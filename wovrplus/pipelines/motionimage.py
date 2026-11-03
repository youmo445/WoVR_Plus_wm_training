"""A separate cached/online path for projected mesh action images."""
import torch
from .world_model import WorldModelPipeline


class MotionImagePipeline(WorldModelPipeline):
    @torch.no_grad()
    def prepare_cached_sample(self, sample):
        tensors = sample["latent_cache"]
        if set(tensors) != {"input_latents", "replay_latents"}:
            raise ValueError("MotionImage requires RGB targets and mesh-motion VAE latents")
        latents = tensors["input_latents"].to(self.device, self.torch_dtype)
        replay = tensors["replay_latents"].to(self.device, self.torch_dtype)
        if replay.shape != latents.shape:
            raise ValueError("Motion latent and RGB latent geometry differ")
        return dict(input_latents=latents, replay_latents=replay,
                    noise=torch.randn_like(latents), context=self.encode_text(sample.get("prompt", "")))

    @torch.no_grad()
    def prepare_sample(self, sample, training=True):
        if "latent_cache" in sample:
            if not training:
                raise ValueError("Training target caches are forbidden during inference")
            return self.prepare_cached_sample(sample)
        if "motion_video" not in sample or len(sample["video"]) != len(sample["motion_video"]):
            raise ValueError("MotionImage requires frame-aligned motion_video")
        latents = self.vae.encode(self.preprocess_video(sample["video"]), device=self.device, tiled=False).to(self.device, self.torch_dtype)
        replay = self.vae.encode(self.preprocess_video(sample["motion_video"]), device=self.device, tiled=False).to(self.device, self.torch_dtype)
        return dict(input_latents=latents, replay_latents=replay, noise=torch.randn_like(latents),
                    context=self.encode_text(sample.get("prompt", "")))
