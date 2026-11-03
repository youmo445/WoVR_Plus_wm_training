"""Dataset adapters. RoboTwin image, vector and mesh-motion conditioning."""

import importlib
import inspect
import json
from pathlib import Path
import numpy as np
import torch

DATASETS = {
    "robotwin": ("robotwin", "RoboTwinActionFKDatasetWancontrolmultiview"),
    "robotwin_endpose": ("robotwin_endpose", "RoboTwinDatasetWancontrolmultiview"),
}


def dataset_class(name):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name}; choices: {list(DATASETS)}")
    module, cls = DATASETS[name]
    return getattr(importlib.import_module("wovrplus.data." + module), cls)


class SampleAdapter(torch.utils.data.Dataset):
    def __init__(self, dataset, cfg):
        self.dataset, self.cfg = dataset, cfg
        self.load_from_cache = False
        for attr in ("sample_weights", "weighted_num_samples"):
            if hasattr(dataset, attr):
                setattr(self, attr, getattr(dataset, attr))
        self.stats = None
        if cfg["data"]["action_stats"]:
            self.stats = json.loads(Path(cfg["data"]["action_stats"]).read_text())
            std = np.asarray(self.stats["std"])
            if not np.isfinite(std).all() or (std <= 0).any():
                raise ValueError("Action normalization std must be finite and positive")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = dict(self.dataset[index])
        sample.setdefault("prompt", "")
        if self.cfg["model"]["family"] == "vector":
            if "action" not in sample:
                if not self.cfg["data"]["name"].startswith("robotwin"):
                    raise ValueError(
                        "This dataset does not expose raw actions; use a RoboTwin adapter for vector"
                    )
                import h5py
                from .robotwin import _read_action_components

                ids = np.asarray(sample["meta"]["frame_ids"])
                unique, inverse = np.unique(ids, return_inverse=True)
                with h5py.File(sample["meta"]["path"], "r") as f:
                    left, lg, right, rg = _read_action_components(f, unique)
                actions = np.concatenate((left, lg[:, None], right, rg[:, None]), axis=1)[inverse]
                sample["action"] = torch.from_numpy(actions)
            action = torch.as_tensor(sample["action"], dtype=torch.float32)
            if action.shape != (len(sample["video"]), self.cfg["model"]["action_dim"]):
                raise ValueError(
                    f'Action/video mismatch {tuple(action.shape)} vs frames={len(sample["video"])} dim={self.cfg["model"]["action_dim"]}'
                )
            if self.stats:
                action = (action - torch.tensor(self.stats["mean"])) / torch.tensor(
                    self.stats["std"]
                )
            sample["action"] = action
        else:
            for key in ("ray_map_o", "ray_map_d"):
                if isinstance(sample.get(key), np.ndarray):
                    sample[key] = torch.from_numpy(sample[key])
        if self.cfg["model"]["family"] == "raymap_vace" and self.cfg["model"]["vace_in_dim"] == 54:
            # The legacy raw-ray variant uses one ray per latent slot; stack4 uses all four.
            for key in ("ray_map_o", "ray_map_d"):
                value = sample.get(key)
                if torch.is_tensor(value) and value.shape[0] == 12:
                    sample[key] = value[:3]
        if self.cfg["model"]["family"] == "motionimage":
            if not hasattr(self, "motion_renderer"):
                from .motionimage_renderer import RoboTwinMeshRenderer
                settings = self.cfg.get("condition_cache", {})
                self.motion_renderer = RoboTwinMeshRenderer(
                    self.cfg["data"]["fk_config"], self.cfg["data"]["resize_to"],
                    settings.get("background", 190), settings.get("gripper_gray", 55))
            sample["motion_video"] = self.motion_renderer.render(sample["meta"])
        return sample


def build_dataset(cfg, split):
    data = cfg["data"]
    mode = data.get("cache", {}).get("mode", "off")
    if mode != "off":
        from .latent_cache import LatentCacheDataset, ReplayCachePlanDataset

        return (LatentCacheDataset if mode == "read" else ReplayCachePlanDataset)(cfg, split)
    name = data["name"]
    cls = dataset_class(name)
    parameters = {}
    for base in reversed(cls.__mro__):
        if base.__module__.startswith("wovrplus.") and "__init__" in base.__dict__:
            parameters.update(inspect.signature(base.__init__).parameters)
    kwargs = dict(data["kwargs"])
    kwargs.update(data.get(split + "_kwargs", {}))
    kwargs.setdefault("base_path", data.get(split + "_path") or data["base_path"])
    kwargs.setdefault("split", split)
    kwargs.setdefault("num_frames", data["context_frames"] + data["horizon_frames"])
    defaults = dict(data["sampling"])
    defaults.update(context_length=data["context_frames"], resize_to=data["resize_to"],
                    view_mode=data["view_mode"], single_camera=data["single_camera"],
                    output_raymap=cfg["model"]["family"] == "raymap_vace" and cfg["model"]["vace_in_dim"] != 48,
                    raymap_mode="image" if cfg["model"]["vace_in_dim"] == 144 else "latent")
    if name != "robotwin_endpose":
        defaults["fk_config_path"] = data["fk_config"]
    for key, value in defaults.items():
        if key in parameters:
            kwargs.setdefault(key, value)
    dataset = cls(**kwargs)
    if len(dataset) == 0:
        raise ValueError(f"{split} dataset is empty")
    return SampleAdapter(dataset, cfg)
