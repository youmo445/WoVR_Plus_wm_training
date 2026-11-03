"""Explicit local WAN loading; no hash-dependent action mode or hidden downloads."""

from pathlib import Path
import torch
from safetensors.torch import load_file
from .utils import init_weights_on_device, load_state_dict
from .wan_dit import WanModel
from .action import WanActionModel
from .wan_vae import WanVideoVAE38
from .wan_text_encoder import WanTextEncoder

WAN_5B = dict(
    dim=3072,
    in_dim=48,
    ffn_dim=14336,
    out_dim=48,
    text_dim=4096,
    freq_dim=256,
    eps=1e-6,
    patch_size=(1, 2, 2),
    num_heads=24,
    num_layers=30,
    has_image_input=False,
    seperated_timestep=True,
    require_clip_embedding=False,
    require_vae_embedding=False,
    fuse_vae_embedding_in_latents=True,
)


def load_wan(model_dir, model_config, dtype=torch.bfloat16):
    root = Path(model_dir)
    shards = sorted(root.glob("diffusion_pytorch_model-*-of-*.safetensors"))
    if not shards:
        shards = [root / "diffusion_pytorch_model.safetensors"]
    state = {}
    for shard in shards:
        for k, v in load_file(str(shard)).items():
            if k in state:
                raise ValueError(f"Duplicate parameter {k} in WAN shards")
            state[k] = v.to(dtype)
    cls, extra = WanModel, {}
    if model_config["family"] == "wan_action":
        cls = WanActionModel
        extra = {k: model_config[k] for k in ("action_dim", "action_mode", "use_text")}
    if model_config["family"] == "wan_physcanvas":
        from .physcanvas import WanPhysCanvasModel
        cls = WanPhysCanvasModel
    with init_weights_on_device():
        dit = cls(**WAN_5B, **extra)
    result = dit.load_state_dict(state, strict=False, assign=True)
    allowed_missing = tuple(
        k for k in result.missing_keys if k.startswith(("action_mlp1.", "action_mlp2.", "replay_embedding."))
    )
    unexpected = [
        k
        for k in result.unexpected_keys
        if not (cls is WanActionModel and not extra["use_text"] and k.startswith("text_embedding."))
    ]
    if set(result.missing_keys) != set(allowed_missing) or unexpected:
        raise ValueError(
            f"Base WAN checkpoint incompatible: missing={result.missing_keys}, unexpected={unexpected}"
        )
    # Only new projections are initialized; pretrained tensors are never reinitialized.
    for name in ("action_mlp1", "action_mlp2", "replay_embedding"):
        if hasattr(dit, name) and any(k.startswith(name + ".") for k in allowed_missing):
            module = getattr(dit, name).to_empty(device="cpu")
            for layer in module.modules():
                if isinstance(layer, (torch.nn.Linear, torch.nn.Conv3d)):
                    layer.reset_parameters()
    dit.to(dtype=dtype)
    vae = load_vae(root, dtype)
    text_encoder = None
    if model_config["family"] == "wan_vace" or model_config["use_text"]:
        state = load_state_dict(str(root / "models_t5_umt5-xxl-enc-bf16.pth"), torch_dtype=dtype)
        with init_weights_on_device():
            text_encoder = WanTextEncoder()
        text_encoder.load_state_dict(state, strict=True, assign=True)
    return dit, vae, text_encoder


def load_vae(model_dir, dtype=torch.bfloat16):
    vae_state = load_state_dict(str(Path(model_dir) / "Wan2.2_VAE.pth"), torch_dtype=dtype)
    vae_state = WanVideoVAE38.state_dict_converter().from_civitai(vae_state)
    with init_weights_on_device():
        vae = WanVideoVAE38()
    vae.load_state_dict(vae_state, strict=True, assign=True)
    return vae.eval().requires_grad_(False)


def load_world_checkpoint(pipe, path):
    """Accept pipe.dit.*, module.pipe.*, dit.* and raw RLinf DiT checkpoints."""
    raw = load_file(str(path))
    state = {}
    for key, value in raw.items():
        if key.startswith("module."):
            key = key[7:]
        if key.startswith("pipe."):
            key = key[5:]
        if not key.startswith(("dit.", "vace.")):
            key = "dit." + key
        state[key] = value
    expected = {k for k in pipe.state_dict() if k.startswith(("dit.", "vace."))}
    if set(state) != expected:
        raise ValueError(
            f"Checkpoint/config mismatch: missing={sorted(expected-set(state))[:12]}, unexpected={sorted(set(state)-expected)[:12]}"
        )
    pipe.load_state_dict(state, strict=False)
