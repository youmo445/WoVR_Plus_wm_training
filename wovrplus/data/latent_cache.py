"""Versioned, deterministic window/augmentation caches. Never cache diffusion noise."""

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from wovrplus.config import ROOT

SCHEMA = 1


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def encoding_spec(cfg):
    data = deepcopy(cfg["data"])
    data.pop("cache", None)
    model = {k: cfg["model"][k] for k in ("family", "dtype", "vace_in_dim", "action_dim")}
    return {"schema": SCHEMA, "data": data, "model": model}


def code_fingerprint():
    return {
        name: digest_file(ROOT / name)
        for name in (
            "wovrplus/models/wan_vae.py",
            "wovrplus/pipelines/base.py",
            "wovrplus/pipelines/vace_conditions.py",
            "wovrplus/data/robotwin.py",
            "wovrplus/data/projection.py",
            "wovrplus/data/registry.py",
            "wovrplus/pipelines/world_model.py",
            "wovrplus/data/latent_cache.py",
        )
    }


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def variant_seed(seed, split, source_index, variant):
    key = f"{seed}:{split}:{source_index}:{variant}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")


@contextmanager
def sample_rng(seed):
    # Seed CPU sampling only. torch.manual_seed() also changes CUDA noise RNG!
    python_state, numpy_state, torch_state = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


def replay_sample(dataset, record):
    with sample_rng(record["seed"]):
        return dataset[record["source_index"]]


def load_manifest(cfg, split):
    root = Path(cfg["data"]["cache"]["root"]).resolve()
    if root.is_relative_to(ROOT.resolve()):
        raise ValueError("Latent cache must be outside the source checkout")
    manifest = json.loads((root / split / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA or not manifest.get("complete"):
        raise ValueError("Incomplete or unsupported latent cache")
    if manifest["encoding_spec"] != encoding_spec(cfg):
        raise ValueError(
            "Latent cache configuration mismatch; regenerate for these data/model settings"
        )
    if manifest["code_sha256"] != code_fingerprint():
        raise ValueError("Latent cache preprocessing code changed; regenerate the cache")
    if manifest["vae_identity"] != file_identity(
        Path(cfg["model"]["model_dir"]) / "Wan2.2_VAE.pth"
    ):
        raise ValueError("Latent cache VAE weights changed")
    for identity in manifest.get("assets", []):
        if file_identity(identity["path"]) != identity:
            raise ValueError(f"Latent cache calibration/normalization changed: {identity['path']}")
    if cfg["data"]["cache"].get("verify_sources", True):
        for identity in manifest["sources"]:
            if file_identity(identity["path"]) != identity:
                raise ValueError(f"Latent cache source changed: {identity['path']}")
    return root / split, manifest


class LatentCacheDataset(torch.utils.data.Dataset):
    load_from_cache = False  # Pipeline still generates fresh noise and text context.

    def __init__(self, cfg, split):
        self.root, self.manifest = load_manifest(cfg, split)
        self.records = self.manifest["records"]
        self.verify_checksum = cfg["data"]["cache"].get("verify_checksum", False)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        path = (self.root / record["file"]).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("Invalid cache record path")
        if self.verify_checksum and digest_file(path) != record["sha256"]:
            raise ValueError(f"Corrupt latent cache: {path}")
        with safe_open(str(path), framework="pt", device="cpu") as f:
            if (f.metadata() or {}).get("signature") != self.manifest["signature"]:
                raise ValueError(f"Cache tensor signature mismatch: {path}")
            if set(f.keys()) != set(record["tensors"]):
                raise ValueError(f"Cache tensor keys mismatch: {path}")
            tensors = {key: f.get_tensor(key) for key in f.keys()}
        for key, value in tensors.items():
            expected = record["tensors"][key]
            if list(value.shape) != expected["shape"] or str(value.dtype) != expected["dtype"]:
                raise ValueError(f"Cache tensor shape/dtype mismatch: {path}:{key}")
        return {
            "latent_cache": tensors,
            "prompt": record["prompt"],
            "meta": record["meta"],
            "cache_record": {"source_index": record["source_index"], "variant": record["variant"]},
        }


class ReplayCachePlanDataset(torch.utils.data.Dataset):
    """Online encoding baseline with the exact same finite augmentations as the cache."""

    load_from_cache = False

    def __init__(self, cfg, split):
        from .registry import build_dataset

        _, self.manifest = load_manifest(cfg, split)
        self.records = self.manifest["records"]
        raw_cfg = deepcopy(cfg)
        raw_cfg["data"]["cache"]["mode"] = "off"
        self.dataset = build_dataset(raw_cfg, split)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return replay_sample(self.dataset, self.records[index])
