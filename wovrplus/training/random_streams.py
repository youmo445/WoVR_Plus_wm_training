"""Independent diffusion RNG streams without changing the data-loader RNG.

The pipeline's existing randn_like call runs inside a short isolated RNG context.
This keeps encoder/cache fingerprints unchanged and avoids generating noise twice.
Timestep draws use a separate explicit CPU generator. Validation never advances
either training generator. There is no model/optimizer offload here.
"""
from contextlib import contextmanager
import hashlib
import torch


def stream_seed(base_seed, rank, phase, domain, round_id=0):
    if phase not in ("train", "validation") or domain not in ("noise", "timestep"):
        raise ValueError("Unknown diffusion RNG stream")
    if rank < 0 or round_id < 0:
        raise ValueError("Rank and round must be nonnegative")
    key = f"wovr-diffusion-v1:{int(base_seed)}:{int(rank)}:{phase}:{domain}:{int(round_id)}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little") % (2**63 - 1)


class DiffusionRandomStreams:
    def __init__(self, base_seed, rank, device, phase="train", round_id=0):
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError(f"Unsupported diffusion RNG device: {self.device}")
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.metadata = dict(
            scheme="rank-phase-domain-v1", base_seed=int(base_seed), rank=int(rank),
            phase=phase, round_id=int(round_id), device=str(self.device),
            noise_seed=stream_seed(base_seed, rank, phase, "noise", round_id),
            timestep_seed=stream_seed(base_seed, rank, phase, "timestep", round_id),
        )
        self.noise_generator = torch.Generator(device=self.device).manual_seed(self.metadata["noise_seed"])
        self.timestep_generator = torch.Generator(device="cpu").manual_seed(self.metadata["timestep_seed"])

    def randint(self, low, high, size=(1,)):
        return torch.randint(low, high, size, generator=self.timestep_generator, device="cpu")

    @contextmanager
    def noise_context(self):
        devices = [self.device.index] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            if self.device.type == "cuda":
                torch.cuda.set_rng_state(self.noise_generator.get_state(), self.device)
            else:
                torch.set_rng_state(self.noise_generator.get_state())
            try:
                yield
            finally:
                state = (torch.cuda.get_rng_state(self.device)
                         if self.device.type == "cuda" else torch.get_rng_state())
                self.noise_generator.set_state(state)
