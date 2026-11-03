import os
import hashlib
import json
import torch
from wovrplus.pipelines.world_model import WorldModelPipeline
from .random_streams import DiffusionRandomStreams


class WorldModelTrainingModule(torch.nn.Module):
    def __init__(self, cfg, pipe=None):
        super().__init__()
        self.cfg = cfg
        pipeline = WorldModelPipeline
        if cfg["model"]["family"] == "wan_physcanvas":
            from wovrplus.pipelines.physcanvas import PhysCanvasPipeline
            pipeline = PhysCanvasPipeline
        self.pipe = pipe or pipeline.from_config(cfg)
        self.pipe.scheduler.set_timesteps(1000, training=True)
        trainable = cfg["training"]["trainable"]
        self.pipe.requires_grad_(False)
        if "dit" in trainable:
            self.pipe.dit.requires_grad_(True)
        if "vace" in trainable:
            if self.pipe.vace is None:
                raise ValueError("vace selected for a direct action model")
            self.pipe.vace.requires_grad_(True)
        if "action" in trainable:
            for name in ("action_mlp1", "action_mlp2"):
                if hasattr(self.pipe.dit, name):
                    getattr(self.pipe.dit, name).requires_grad_(True)
        if not any(p.requires_grad for p in self.parameters()):
            raise ValueError("No trainable parameters selected")
        self.diffusion_streams = None
        self.diffusion_train_draws = 0

    def configure_diffusion_rng(self, seed, rank, device):
        self.diffusion_seed, self.diffusion_rank = int(seed), int(rank)
        self.diffusion_device = torch.device(device)
        self.diffusion_streams = {
            phase: DiffusionRandomStreams(seed, rank, device, phase)
            for phase in ("train", "validation")
        }
        return {phase: rng.metadata for phase, rng in self.diffusion_streams.items()}

    def begin_evaluation(self, round_id):
        self.diffusion_streams["validation"] = DiffusionRandomStreams(
            self.diffusion_seed, self.diffusion_rank, self.diffusion_device,
            "validation", round_id=round_id)

    @property
    def last_loss_metrics(self):
        return self.pipe.last_loss_metrics

    def to(self, *args, **kwargs):
        self.pipe.to(*args, **kwargs)
        return self

    def train(self, mode=True):
        super().train(mode)
        # Frozen encoder dropout must not be re-enabled by outer model.train().
        self.pipe.vae.eval()
        if self.pipe.text_encoder is not None:
            self.pipe.text_encoder.eval()
        return self

    def trainable_modules(self):
        return (p for p in self.parameters() if p.requires_grad)

    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        # Store all trainable-family model weights, including a frozen DiT when adapter-only tuning.
        # This keeps a checkpoint self-contained relative to the frozen base VAE/text encoder.
        return {
            k: v.contiguous()
            for k, v in state_dict.items()
            if k.startswith(("pipe.dit.", "pipe.vace."))
        }

    def forward(self, data, inputs=None):
        if self.diffusion_streams is None:
            self.configure_diffusion_rng(self.cfg["training"].get("seed", 42),
                                         int(os.environ.get("RANK", "0")), self.pipe.device)
        phase = "train" if self.training else "validation"
        rng = self.diffusion_streams[phase]
        if inputs is None:
            with rng.noise_context():
                inputs = self.pipe.prepare_sample(data)
        else:
            # Precomputed *inputs* may never prescribe diffusion noise/timesteps.
            inputs = dict(inputs)
            with rng.noise_context():
                inputs["noise"] = torch.randn_like(inputs["input_latents"])
        t = self.cfg["training"]
        loss = self.pipe.training_loss(
            dit=self.pipe.dit,
            vace=self.pipe.vace,
            use_gradient_checkpointing=t["gradient_checkpointing"],
            use_gradient_checkpointing_offload=t["gradient_checkpointing_offload"],
            clean_latent_slots=(self.cfg["data"]["context_frames"] + 3) // 4,
            ti2v2_low_noise_step_offset=t["low_noise_offset"],
            ti2v2_horizon_loss_latent_slots=self.cfg["data"]["horizon_frames"] // 4,
            diffusion_rng=rng,
            **inputs
        )
        if self.training:
            self.diffusion_train_draws += 1
            if self.diffusion_train_draws <= 3:
                probe = inputs["noise"].detach().reshape(-1)[:64].float().cpu().numpy().tobytes()
                print("DIFFUSION_RNG_DRAW " + json.dumps(dict(
                    rank=self.diffusion_rank, microstep=self.diffusion_train_draws,
                    timestep_id=self.last_loss_metrics["timestep_id"],
                    noise_sha256=hashlib.sha256(probe).hexdigest())), flush=True)
        return loss


class CheckpointWriter:
    def __init__(self, output_path):
        self.output_path = output_path

    def save_model(self, accelerator, model, file_name):
        accelerator.wait_for_everyone()
        # get_state_dict can contain collective operations under ZeRO; all ranks participate.
        state = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            state = accelerator.unwrap_model(model).export_trainable_state_dict(state)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            temp = path + ".tmp"
            accelerator.save(state, temp, safe_serialization=True)
            os.replace(temp, path)
        accelerator.wait_for_everyone()
