"""PhysCanvas action-only adaptation: replay patches added to WAN patches.

Reference: PhysCanvas/models/transformer_wan_rnd_mix_stage_one.py (Apache-2.0).
This independent adaptation retains our WAN backbone and training objective;
it does not introduce tactile/state canvases or reference-memory extensions.
"""
from torch import nn
from .wan_dit import WanModel


class WanPhysCanvasModel(WanModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.replay_embedding = nn.Conv3d(
            self.patch_embedding.in_channels, self.dim,
            kernel_size=self.patch_size, stride=self.patch_size,
        )

