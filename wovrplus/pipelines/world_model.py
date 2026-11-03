"""WAN 2.2 world-model components and chunk generation."""

import types
import torch
from .base import BasePipeline
from .model_fn import model_fn_wan
from .vace_conditions import VACEConditionEncoder
from wovrplus.models.loading import load_wan, load_world_checkpoint
from wovrplus.models.vace import VaceWanModel
from wovrplus.schedulers.flow_match import FlowMatchScheduler
from wovrplus.training.loss import world_model_loss


class WorldModelPipeline(BasePipeline):
    def __init__(
        self, dit, vae, text_encoder=None, vace=None, tokenizer=None, dtype=torch.bfloat16
    ):
        super().__init__(
            device="cpu",
            torch_dtype=dtype,
            height_division_factor=32,
            width_division_factor=32,
            time_division_factor=4,
            time_division_remainder=1,
        )
        self.dit, self.vae, self.text_encoder, self.vace = dit, vae, text_encoder, vace
        self.tokenizer = tokenizer
        self.condition_encoder = VACEConditionEncoder()
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.model_fn = model_fn_wan
        self.training_loss = types.MethodType(world_model_loss, self)

    @classmethod
    def from_config(cls, cfg):
        m = cfg["model"]
        dtype = getattr(torch, m["dtype"])
        dit, vae, text_encoder = load_wan(m["model_dir"], m, dtype)
        vace = None
        if m["family"] == "wan_vace":
            vace = VaceWanModel(
                vace_in_dim=m["vace_in_dim"], view_embedding_num_views=m["view_embeddings"]
            ).to(dtype)
            if not m["vace_random_init"]:
                vace.init_from_dit(dit)
        tokenizer = None
        if text_encoder is not None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(m["tokenizer_path"], local_files_only=True)
        pipe = cls(dit, vae, text_encoder, vace, tokenizer, dtype)
        if m["checkpoint"]:
            load_world_checkpoint(pipe, m["checkpoint"])
        return pipe

    @torch.no_grad()
    def encode_text(self, prompt):
        if self.text_encoder is None:
            return None
        import ftfy, html, re

        with self.profile_scope("Text/clean_and_tokenize"):
            prompt = re.sub(
                r"\s+", " ", html.unescape(html.unescape(ftfy.fix_text(prompt)))
            ).strip()
            tokens = self.tokenizer(
                [prompt],
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt",
                add_special_tokens=True,
            )
        with self.profile_scope("Text/token_host_to_device"):
            ids, mask = tokens.input_ids.to(self.device), tokens.attention_mask.to(self.device)
        context = self.text_encoder(ids, mask)
        with self.profile_scope("Text/mask_padding"):
            for i, n in enumerate(mask.gt(0).sum(dim=1)):
                context[i, n:] = 0
        return context

    @torch.no_grad()
    def prepare_sample(self, sample, training=True):
        if "latent_cache" in sample:
            if not training:
                raise ValueError(
                    "Training latent caches contain future targets; inference must use raw context"
                )
            return self.prepare_cached_sample(sample)
        video = sample["video"]
        if len(video) % 4 != 1:
            raise ValueError("WAN video length must be 4n+1")
        height, width = video[0].size[1], video[0].size[0]
        if height % 32 or width % 32:
            raise ValueError("Resize each camera so stacked height and width are divisible by 32")
        with self.profile_scope("Input/video_preprocess"):
            video_tensor = self.preprocess_video(video)
        latents = self.vae.encode(video_tensor, device=self.device, tiled=False).to(
            self.device, self.torch_dtype
        )
        prepared = {
            "input_latents": latents,
            "noise": torch.randn_like(latents),
            "context": self.encode_text(sample.get("prompt", "")),
        }
        if self.vace is not None:
            if not any(
                sample.get(k) is not None for k in ("action_map", "vace_video", "control_video")
            ):
                raise ValueError("VACE dataset must provide action/control maps")
            prepared.update(
                self.condition_encoder.process(
                    self,
                    sample.get("vace_video", sample.get("control_video")),
                    sample.get("action_map"),
                    sample.get("ray_map_o"),
                    sample.get("ray_map_d"),
                    1.0,
                    height,
                    width,
                    len(video),
                    False,
                    (34, 34),
                    (18, 16),
                )
            )
        else:
            if "action" not in sample:
                raise ValueError("Dataset must provide action tensors for wan_action")
            prepared["action"] = sample["action"]
        return prepared

    @torch.no_grad()
    def prepare_cached_sample(self, sample):
        tensors = sample["latent_cache"]
        required = {"input_latents", "vace_context" if self.vace is not None else "action"}
        if set(tensors) != required:
            raise ValueError(f"Cached sample requires exactly {required}")
        with self.profile_scope("Cache/latents_to_device"):
            latents = tensors["input_latents"].to(self.device, self.torch_dtype)
            prepared = {"input_latents": latents}
            if self.vace is not None:
                control = tensors["vace_context"].to(self.device, self.torch_dtype)
                if (
                    control.shape[1] != self.vace.vace_in_dim
                    or control.shape[2:] != latents.shape[2:]
                ):
                    raise ValueError("Cached VACE condition shape does not match model/target")
                prepared.update(vace_context=control, vace_global_context=None, vace_scale=1.0)
            else:
                prepared["action"] = tensors["action"].to(self.device)
        # Neither diffusion noise nor timestep may be cached.
        prepared["noise"] = torch.randn_like(latents)
        prepared["context"] = self.encode_text(sample.get("prompt", ""))
        return prepared

    @torch.no_grad()
    def generate(self, sample, context_frames=5, steps=50, seed=42, cfg_scale=1.0):
        """Generate one chunk. Input video supplies only context; future slots are zeroed before VAE encoding."""
        if "latent_cache" in sample:
            raise ValueError(
                "Training latent cache cannot be used for inference (future target leakage)"
            )
        if context_frames not in (1, 5):
            raise ValueError("context_frames must be 1 or 5")
        from PIL import Image

        if context_frames >= len(sample["video"]):
            raise ValueError("Need future frames to predict")
        sample = dict(sample)
        sample["video"] = list(sample["video"][:context_frames]) + [
            Image.new("RGB", sample["video"][0].size)
        ] * (len(sample["video"]) - context_frames)
        inputs = self.prepare_sample(sample, training=False)
        fixed = inputs.pop("input_latents")
        inputs.pop("noise")
        clean = (context_frames + 3) // 4
        latents = self.generate_noise(fixed.shape, seed=seed, rand_device="cpu")
        latents[:, :, :clean] = fixed[:, :, :clean]
        was_training = self.scheduler.training
        self.scheduler.set_timesteps(steps, training=False)
        try:
            for timestep in self.scheduler.timesteps:
                t = timestep.reshape(1).to(self.device, self.torch_dtype)
                pred = self.model_fn(
                    self.dit, latents, t, vace=self.vace, clean_latent_slots=clean, **inputs
                )
                if cfg_scale != 1:
                    negative = dict(inputs)
                    if "action" in negative:
                        negative["action"] = torch.zeros_like(torch.as_tensor(negative["action"]))
                    if negative.get("context") is not None:
                        negative["context"] = self.encode_text("")
                    uncond = self.model_fn(
                        self.dit, latents, t, vace=self.vace, clean_latent_slots=clean, **negative
                    )
                    pred = uncond + cfg_scale * (pred - uncond)
                latents = self.scheduler.step(pred, timestep, latents)
                latents[:, :, :clean] = fixed[:, :, :clean]
            decoded = self.vae.decode(latents, device=self.device)
            return self.vae_output_to_video(decoded)
        finally:
            if was_training:
                self.scheduler.set_timesteps(1000, training=True)
