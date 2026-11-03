"""Component caches reuse audited RGB latents without weakening legacy checks."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from wovrplus.config import ROOT
from .latent_cache import load_manifest, digest_file, file_identity


def motion_key(meta: dict) -> str:
    payload = {k: meta[k] for k in ("path", "frame_ids", "camera_names")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def episode_key(path: str) -> str:
    return hashlib.sha256(path.encode()).hexdigest()


def source_manifest(cfg: dict, split: str):
    """Only family differs; validate data, VAE and all encoder source hashes."""
    original = deepcopy(cfg)
    original["model"]["family"] = "raymap_vace"
    if original["model"]["vace_in_dim"] != 72:
        raise ValueError("Component reuse is audited only for the original VACE72 cache")
    return load_manifest(original, split)


def condition_spec(cfg: dict, manifest: dict) -> dict:
    settings = {k: v for k, v in cfg["condition_cache"].items() if k not in ("root", "allow_partial_for_smoke")}
    names = ["wovrplus/data/condition_cache.py", "wovrplus/data/build_condition_cache.py"]
    if settings["kind"] == "motionimage":
        names += ["wovrplus/data/motionimage_renderer.py"]
    return dict(schema=1, source_signature=manifest["signature"], settings=settings,
                code_sha256={name: digest_file(ROOT / name) for name in names})


class ConditionCacheDataset(torch.utils.data.Dataset):
    load_from_cache = False  # Fresh diffusion noise/timestep, just as the original cache.

    def __init__(self, cfg, split):
        self.source_root, self.source = source_manifest(cfg, split)
        self.root = Path(cfg["condition_cache"]["root"]).resolve() / split
        if self.root.is_relative_to(ROOT.resolve()):
            raise ValueError("Condition data must remain outside the checkout")
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest["spec"] != condition_spec(cfg, self.source):
            raise ValueError("Condition cache settings/source/code changed")
        for asset in self.manifest.get("assets", []):
            if file_identity(asset["path"]) != asset:
                raise ValueError(f"Condition geometry asset changed: {asset['path']}")
        self.kind = cfg["condition_cache"]["kind"]
        self.entries = self.manifest["entries"]
        self.records = self.source["records"]
        if not self.manifest["complete"]:
            if not cfg["condition_cache"].get("allow_partial_for_smoke") or not 0 < cfg["training"]["max_steps"] <= 3:
                raise ValueError("Condition cache is incomplete")
            self.records = [r for r in self.records if self.key(r) in self.entries]
        if not self.records:
            raise ValueError("Empty condition dataset")
        if any(self.key(r) not in self.entries for r in self.records):
            raise ValueError("Condition manifest does not cover all RGB cache records")
        self.verify_checksum = cfg["data"]["cache"].get("verify_checksum", False)

    def key(self, record):
        return motion_key(record["meta"]) if self.kind == "motionimage" else episode_key(record["meta"]["path"])

    def __len__(self):
        return len(self.records)

    @lru_cache(maxsize=512)
    def actions(self, key):
        entry = self.entries[key]
        path = self.root / entry["file"]
        if digest_file(path) != entry["sha256"]:
            raise ValueError(f"Action array checksum mismatch: {path}")
        return np.load(path, mmap_mode="r", allow_pickle=False)

    def __getitem__(self, index):
        record = self.records[index]
        path = (self.source_root / record["file"]).resolve()
        if not path.is_relative_to(self.source_root.resolve()):
            raise ValueError("Cache record escaped its root")
        if self.verify_checksum and digest_file(path) != record["sha256"]:
            raise ValueError("RGB cache checksum mismatch")
        with safe_open(str(path), framework="pt", device="cpu") as stream:
            if (stream.metadata() or {}).get("signature") != self.source["signature"]:
                raise ValueError("RGB cache signature mismatch")
            rgb = stream.get_tensor("input_latents")
        expected = record["tensors"]["input_latents"]
        if list(rgb.shape) != expected["shape"] or str(rgb.dtype) != expected["dtype"]:
            raise ValueError("RGB cache shape/dtype mismatch")
        key = self.key(record)
        tensors = {"input_latents": rgb}
        if self.kind == "vector":
            values = self.actions(key)[record["meta"]["frame_ids"]].copy()
            tensors["action"] = torch.from_numpy(values)
        else:
            entry = self.entries[key]
            path = self.root / entry["file"]
            if self.verify_checksum and digest_file(path) != entry["sha256"]:
                raise ValueError("Motion cache checksum mismatch")
            with safe_open(str(path), framework="pt", device="cpu") as stream:
                if (stream.metadata() or {}).get("signature") != self.manifest["signature"] or stream.metadata().get("motion_key") != key:
                    raise ValueError("Motion cache signature/key mismatch")
                replay = stream.get_tensor("replay_latents")
            if replay.shape != rgb.shape or replay.dtype != rgb.dtype:
                raise ValueError("Motion cache shape/dtype mismatch")
            tensors["replay_latents"] = replay
        return dict(latent_cache=tensors, prompt=record["prompt"], meta=record["meta"],
                    cache_record={"source_index": record["source_index"], "variant": record["variant"]})


def build_dataset(cfg, split):
    if cfg.get("condition_cache"):
        if cfg["data"]["cache"]["mode"] != "read":
            raise ValueError("Condition cache training requires data.cache.mode=read")
        expected = {"vector": "vector", "motionimage": "motionimage"}
        if expected.get(cfg["model"]["family"]) != cfg["condition_cache"]["kind"]:
            raise ValueError("Condition representation does not match model family")
        if cfg["data"]["action_stats"]:
            raise ValueError("This joint14 cache preserves raw actions; normalization needs a new cache spec")
        return ConditionCacheDataset(cfg, split)
    from .registry import build_dataset as original
    return original(cfg, split)
