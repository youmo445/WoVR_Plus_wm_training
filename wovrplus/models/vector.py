"""Action projections adapted from youmo445/diffsynth-studio-rlinf (Apache-2.0).

Keep action_mlp1/action_mlp2 on the DiT so existing parameter names stay compatible.
AdaLN uses leading zero padding and contiguous groups of four actions, as upstream.
"""

import torch
from torch import nn
from .wan_dit import WanModel

ACTION_MODES = {
    "cross_attn": "crossattn",
    "cross-attn": "crossattn",
    "crossattn": "crossattn",
    "adaln": "modulation",
    "modulation": "modulation",
    "cross_attn_and_adaln": "both",
    "cross-attn_and_adaln": "both",
    "both": "both",
}


def normalize_action_mode(mode):
    try:
        return ACTION_MODES[mode.lower()]
    except (KeyError, AttributeError):
        raise ValueError(
            f"Unknown action mode {mode!r}; choose cross_attn, adaln, cross_attn_and_adaln"
        ) from None


class WanVectorModel(WanModel):
    def __init__(self, *args, action_dim=14, action_mode="adaln", use_text=False, **kwargs):
        super().__init__(*args, **kwargs)
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        self.action_dim = action_dim
        self.action_mode = normalize_action_mode(action_mode)
        self.use_text = use_text
        self.enable_action_crossattn = self.action_mode in ("crossattn", "both")
        self.enable_action_modulation = self.action_mode in ("modulation", "both")
        if not use_text:
            del self.text_embedding
        if self.enable_action_crossattn:
            self.action_mlp1 = nn.Sequential(
                nn.Linear(action_dim, self.dim), nn.GELU(), nn.Linear(self.dim, self.dim)
            )
        if self.enable_action_modulation:
            self.action_mlp2 = nn.Sequential(
                nn.Linear(action_dim * 4, self.dim * 4),
                nn.SiLU(),
                nn.Linear(self.dim * 4, self.dim),
            )

    def action_embeddings(self, action, latent_frames, spatial_tokens):
        param = next(self.parameters())
        action = torch.as_tensor(action, device=param.device, dtype=param.dtype)
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3 or action.shape[-1] != self.action_dim:
            raise ValueError(f"Expected action [B,T,{self.action_dim}], got {tuple(action.shape)}")
        if action.shape[1] == 0:
            raise ValueError("Action sequence cannot be empty")
        tokens = self.action_mlp1(action) if self.enable_action_crossattn else None
        modulation = None
        if self.enable_action_modulation:
            padding = (-action.shape[1]) % 4
            if padding:
                action = torch.cat(
                    (action.new_zeros(action.shape[0], padding, self.action_dim), action), dim=1
                )
            grouped = action.reshape(action.shape[0], -1, self.action_dim * 4)
            if grouped.shape[1] != latent_frames:
                raise ValueError(
                    f"Action groups {grouped.shape[1]} != video latent frames {latent_frames}; align sampled actions with video frames"
                )
            modulation = self.action_mlp2(grouped).repeat_interleave(spatial_tokens, dim=1)
        return tokens, modulation
