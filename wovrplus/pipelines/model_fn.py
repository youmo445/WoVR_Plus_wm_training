"""Shared WAN forward for VACE controls or RLinf-compatible action injection."""

from contextlib import nullcontext
import torch
from torch.utils.checkpoint import checkpoint
from einops import rearrange
from wovrplus.models.wan_dit import sinusoidal_embedding_1d


def model_fn_wan(
    dit,
    latents,
    timestep,
    context=None,
    action=None,
    vace=None,
    vace_context=None,
    vace_scale=1.0,
    replay_latents=None,
    clean_latent_slots=2,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    **kwargs,
):
    if latents.ndim != 5:
        raise ValueError("latents must be [B,C,F,H,W]")
    b, _, f, h0, w0 = latents.shape
    pf, ph, pw = dit.patch_size
    if pf != 1 or h0 % ph or w0 % pw:
        raise ValueError(
            "World-model forward requires temporal patch 1 and divisible spatial sizes"
        )
    if not 0 <= clean_latent_slots < f:
        raise ValueError("clean_latent_slots must leave at least one future latent slot")
    h, w = h0 // ph, w0 // pw
    time = torch.as_tensor(timestep, device=latents.device, dtype=latents.dtype).reshape(-1)
    if time.numel() == 1:
        time = time.expand(b)
    if time.numel() != b:
        raise ValueError("timestep must be scalar or one value per batch item")
    times = time[:, None, None].expand(b, f, h * w).clone()
    times[:, :clean_latent_slots] = 0
    t = dit.time_embedding(
        sinusoidal_embedding_1d(dit.freq_dim, times.flatten()).to(latents.dtype)
    ).reshape(b, f * h * w, dit.dim)
    if context is not None:
        if not hasattr(dit, "text_embedding"):
            raise ValueError("Text context supplied to an action-only model; enable model.use_text")
        context = dit.text_embedding(context)
    if hasattr(dit, "action_mode"):
        if action is None:
            raise ValueError("Direct action model requires aligned action tensors")
        action_context, action_mod = dit.action_embeddings(action, f, h * w)
        action_batch = (action_context if action_context is not None else action_mod).shape[0]
        if action_batch != b:
            raise ValueError(f"Action batch {action_batch} != latent batch {b}")
        if action_context is not None:
            context = (
                action_context if context is None else torch.cat((context, action_context), dim=1)
            )
        if action_mod is not None:
            t = t + action_mod
    elif action is not None:
        raise ValueError("Use a WanVectorModel for raw action injection")
    if context is None:
        # Preserve the modulation-only RLinf path: a single zero context token.
        context = latents.new_zeros(b, 1, dit.dim)
    t_mod = dit.time_projection(t).unflatten(-1, (6, dit.dim))
    patches = dit.patchify(latents)
    if hasattr(dit, "replay_embedding"):
        if replay_latents is None or replay_latents.shape != latents.shape:
            raise ValueError("MotionImage requires aligned 48-channel replay latents")
        patches = patches + dit.replay_embedding(replay_latents)
    elif replay_latents is not None:
        raise ValueError("Replay conditions require WanMotionImageModel")
    x = rearrange(patches, "b c f h w -> b (f h w) c").contiguous()
    freqs = (
        torch.cat(
            [
                dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        )
        .reshape(f * h * w, 1, -1)
        .to(x.device)
    )
    hints = None
    if vace_context is not None:
        if vace is None:
            raise ValueError("VACE controls supplied without a VACE branch")
        hints = vace(
            x,
            vace_context,
            context,
            t_mod,
            freqs,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
    for index, block in enumerate(dit.blocks):
        if use_gradient_checkpointing or use_gradient_checkpointing_offload:
            with (
                torch.autograd.graph.save_on_cpu()
                if use_gradient_checkpointing_offload
                else nullcontext()
            ):
                x = checkpoint(block, x, context, t_mod, freqs, use_reentrant=False)
        else:
            x = block(x, context, t_mod, freqs)
        if hints is not None and index in vace.vace_layers_mapping:
            x = x + hints[vace.vace_layers_mapping[index]] * vace_scale
    return dit.unpatchify(dit.head(x, t), (f, h, w))
