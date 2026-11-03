import torch
from .base import PipelineUnit


class VACEConditionEncoder(PipelineUnit):
    """Build Wan2.2 VACE controls from ray/action maps.

    Compatible modes:
    - vace_in_dim=48: VAE-encode only action_map/vace_video into 48 channels.
    - vace_in_dim=144: VAE-encode ray_map_o, ray_map_d, and action_map/vace_video
      into 48 channels each.
    - vace_in_dim=54: VAE-encode only action_map/vace_video into 48 channels,
      then concatenate raw float ray_map_o and ray_map_d tensors with 3 channels each.
    - vace_in_dim=72: same as 54, but each ray tensor stacks four source frames
      per Wan VAE temporal slot, so ray_map_o and ray_map_d have 12 channels each.
    """

    def __init__(self):
        super().__init__(
            input_params=(
                "vace_video",
                "action_map",
                "ray_map_o",
                "ray_map_d",
                "vace_scale",
                "height",
                "width",
                "num_frames",
                "tiled",
                "tile_size",
                "tile_stride",
            ),
            onload_model_names=("vae",),
        )

    def _as_latents(self, pipe, value, name, tiled, tile_size, tile_stride):
        if value is None:
            raise ValueError(f"{name} is required for Wan2.2 VACE.")
        if torch.is_tensor(value):
            latents = value.to(dtype=pipe.torch_dtype, device=pipe.device)
            if latents.ndim == 4:
                latents = latents.unsqueeze(0)
            if latents.ndim != 5:
                raise ValueError(
                    f"{name} tensor must have 4 or 5 dims, got {tuple(latents.shape)}."
                )
            return latents
        with pipe.profile_scope("Input/" + name.replace("/", "_") + "_preprocess"):
            video = pipe.preprocess_video(value)
        return pipe.vae.encode(
            video,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)

    def _as_raw_ray_latents(self, pipe, value, name, expected_t_hw, expected_channels):
        if value is None:
            raise ValueError(f"{name} is required for Wan2.2 VACE raw-ray mode.")
        if not torch.is_tensor(value):
            raise ValueError(
                f"{name} must be a raw float tensor for raw-ray mode, got {type(value)!r}."
            )
        latents = value.to(dtype=pipe.torch_dtype, device=pipe.device)
        if latents.ndim == 4:
            latents = latents.unsqueeze(0)
        if latents.ndim != 5:
            raise ValueError(f"{name} tensor must have 4 or 5 dims, got {tuple(latents.shape)}.")
        if latents.shape[1] != expected_channels:
            raise ValueError(
                f"{name} must have {expected_channels} channels in raw-ray mode, got {latents.shape[1]}."
            )
        if tuple(latents.shape[2:]) != tuple(expected_t_hw):
            raise ValueError(
                f"{name} shape {tuple(latents.shape[2:])} must match action latent shape {tuple(expected_t_hw)}."
            )
        return latents

    def process(
        self,
        pipe,
        vace_video,
        action_map,
        ray_map_o,
        ray_map_d,
        vace_scale,
        height,
        width,
        num_frames,
        tiled,
        tile_size,
        tile_stride,
    ):
        del height, width, num_frames
        action_source = action_map if action_map is not None else vace_video
        if action_source is None and ray_map_o is None and ray_map_d is None:
            return {"vace_context": None, "vace_global_context": None, "vace_scale": vace_scale}
        if action_source is None:
            raise ValueError("Wan2.2 VACE requires action_map or vace_video.")

        pipe.load_models_to_device(["vae"])
        expected_in_dim = getattr(pipe.vace, "vace_in_dim", 144)
        if expected_in_dim != 48 and (ray_map_o is None or ray_map_d is None):
            raise ValueError(
                "Wan2.2 VACE ray-map modes require ray_map_o, ray_map_d, "
                "and action_map or vace_video."
            )
        action_latents = self._as_latents(
            pipe, action_source, "action_map/vace_video", tiled, tile_size, tile_stride
        )
        z_dim = getattr(pipe.vae, "z_dim", getattr(getattr(pipe.vae, "model", None), "z_dim", 48))
        if action_latents.shape[1] != z_dim:
            raise ValueError(
                f"action_map/vace_video should have {z_dim} channels after VAE encode, got {action_latents.shape[1]}."
            )

        if expected_in_dim == z_dim:
            vace_context = action_latents
        elif expected_in_dim in (z_dim + 6, z_dim + 24):
            expected_ray_channels = 3 if expected_in_dim == z_dim + 6 else 12
            ray_o_latents = self._as_raw_ray_latents(
                pipe, ray_map_o, "ray_map_o", action_latents.shape[2:], expected_ray_channels
            )
            ray_d_latents = self._as_raw_ray_latents(
                pipe, ray_map_d, "ray_map_d", action_latents.shape[2:], expected_ray_channels
            )
            vace_context = torch.cat((action_latents, ray_o_latents, ray_d_latents), dim=1)
        elif expected_in_dim == z_dim * 3:
            ray_o_latents = self._as_latents(
                pipe, ray_map_o, "ray_map_o", tiled, tile_size, tile_stride
            )
            ray_d_latents = self._as_latents(
                pipe, ray_map_d, "ray_map_d", tiled, tile_size, tile_stride
            )
            shapes = {tuple(u.shape[2:]) for u in (ray_o_latents, ray_d_latents, action_latents)}
            if len(shapes) != 1:
                raise ValueError(
                    "ray/action latent temporal-spatial shapes must match, got "
                    f"{ray_o_latents.shape}, {ray_d_latents.shape}, {action_latents.shape}."
                )
            for name, latents in (("ray_map_o", ray_o_latents), ("ray_map_d", ray_d_latents)):
                if latents.shape[1] != z_dim:
                    raise ValueError(
                        f"{name} should have {z_dim} channels after VAE encode, got {latents.shape[1]}."
                    )
            vace_context = torch.cat((ray_o_latents, ray_d_latents, action_latents), dim=1)
        else:
            raise ValueError(
                f"Unsupported VACE input dim {expected_in_dim}; expected "
                f"{z_dim}, {z_dim + 6}, {z_dim + 24}, or {z_dim * 3}."
            )

        if vace_context.shape[1] != expected_in_dim:
            raise ValueError(
                f"Expected VACE input dim {expected_in_dim}, got {vace_context.shape[1]}."
            )
        return {"vace_context": vace_context, "vace_global_context": None, "vace_scale": vace_scale}
