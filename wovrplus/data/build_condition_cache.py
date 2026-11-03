"""Build immutable action conditions aligned to an existing RGB latent manifest."""
import argparse
from datetime import timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np
import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import save_file

from wovrplus.config import resolve_config
from .condition_cache import condition_spec, source_manifest, motion_key, episode_key
from .latent_cache import digest_file, file_identity
from .robotwin import _action_source_length, _read_action_components


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    temporary.replace(path)


def build(cfg: dict, limit: int = 0) -> None:
    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world > 1
    if distributed:
        dist.init_process_group("gloo", timeout=timedelta(hours=95))
    torch.set_num_threads(4)
    kind = cfg["condition_cache"]["kind"]
    renderer, vae = None, None
    root = Path(cfg["condition_cache"]["root"])
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / ".build.lock").open("a") if rank == 0 else None
    if lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if distributed:
        dist.barrier()
    try:
        if kind == "motionimage":
            from .motionimage_renderer import RoboTwinMeshRenderer
            from wovrplus.models.loading import load_vae
            renderer = RoboTwinMeshRenderer(cfg["data"]["fk_config"], cfg["data"]["resize_to"], cfg["condition_cache"]["background"], cfg["condition_cache"]["gripper_gray"])
            device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
            torch.cuda.set_device(device)
            vae = load_vae(cfg["model"]["model_dir"]).to(device)
        elif kind != "vector":
            raise ValueError(f"Unknown condition cache kind: {kind}")
        for split in ("train", "val"):
            _, source = source_manifest(cfg, split)
            folder = root / split
            (folder / "tensors").mkdir(parents=True, exist_ok=True)
            spec = condition_spec(cfg, source)
            assets = [file_identity(path) for path in dict.fromkeys(renderer.assets)] if renderer else []
            signature = hashlib.sha256(json.dumps({"spec": spec, "assets": assets}, sort_keys=True).encode()).hexdigest()
            unique = {}
            for record in source["records"]:
                key = motion_key(record["meta"]) if kind == "motionimage" else episode_key(record["meta"]["path"])
                unique.setdefault(key, record)
            selected = list(unique)[:limit or None]
            if (folder / "manifest.json").exists():
                previous = json.loads((folder / "manifest.json").read_text())
                if previous["signature"] != signature:
                    raise ValueError("Existing condition cache has different provenance; use a new root")
                if previous["complete"] and len(previous["entries"]) == len(unique):
                    print(f"CONDITION_CACHE_COMPLETE split={split} entries={len(unique)} reused=true", flush=True)
                    continue
            entries = {}
            started = time.monotonic()
            subset = selected[rank::world]
            for count, key in enumerate(subset, 1):
                record = unique[key]
                path = folder / "tensors" / (key + (".safetensors" if vae else ".npy"))
                if vae is None:
                    with h5py.File(record["meta"]["path"], "r") as source_file:
                        left, lg, right, rg = _read_action_components(source_file, np.arange(_action_source_length(source_file)))
                    actions = np.concatenate((left, lg[:, None], right, rg[:, None]), axis=1).astype(np.float32)
                    if actions.ndim != 2 or actions.shape[1] != 14 or not np.isfinite(actions).all():
                        raise ValueError("Invalid raw joint14 actions")
                    with path.with_suffix(".tmp").open("wb") as stream:
                        np.save(stream, actions, allow_pickle=False)
                    path.with_suffix(".tmp").replace(path)
                    shape = list(actions.shape)
                else:
                    shape = record["tensors"]["input_latents"]["shape"]
                    if not path.exists():
                        images = renderer.render(record["meta"])
                        array = np.stack([np.asarray(image) for image in images])
                        if rank == 0 and count == 1:
                            images[-1].save(folder / "motion-preview.png")
                        video = torch.from_numpy(array).permute(3, 0, 1, 2).unsqueeze(0).to(device=device, dtype=torch.bfloat16) / 127.5 - 1
                        with torch.inference_mode():
                            encoded = vae.encode(video, device=device, tiled=False).cpu().contiguous()
                        if list(encoded.shape) != shape or not torch.isfinite(encoded).all():
                            raise ValueError("Invalid PhysCanvas VAE encoding")
                        save_file({"replay_latents": encoded}, str(path.with_suffix(".tmp")), metadata={"signature": signature, "motion_key": key})
                        path.with_suffix(".tmp").replace(path)
                    with safe_open(str(path), framework="pt", device="cpu") as stream:
                        if stream.metadata() != {"signature": signature, "motion_key": key} or list(stream.get_slice("replay_latents").get_shape()) != shape:
                            raise ValueError(f"Invalid resumed condition tensor: {path}")
                entries[key] = dict(file=str(path.relative_to(folder)), sha256=digest_file(path), shape=shape, bytes=path.stat().st_size)
                if count % 50 == 0 or count == len(subset):
                    print("CONDITION_PROGRESS " + json.dumps(dict(split=split, rank=rank, done=count, total=len(subset), elapsed_seconds=time.monotonic()-started)), flush=True)
            atomic_json(folder / f"rank-{rank}.json", dict(signature=signature, entries=entries))
            if distributed:
                dist.barrier()
            if rank == 0:
                merged = {}
                for worker in range(world):
                    shard = json.loads((folder / f"rank-{worker}.json").read_text())
                    if shard["signature"] != signature or set(merged) & set(shard["entries"]):
                        raise ValueError("Invalid condition-cache shard provenance or overlap")
                    merged.update(shard["entries"])
                if set(merged) != set(selected):
                    raise ValueError("Condition cache coverage audit failed")
                complete = set(merged) == set(unique)
                atomic_json(folder / "manifest.json", dict(schema=1, signature=signature, spec=spec, assets=assets, complete=complete, entries=merged, source_records=len(source["records"]), unique_conditions=len(unique)))
                print(f"CONDITION_CACHE_AUDIT split={split} complete={complete} conditions={len(merged)} source_records={len(source['records'])}", flush=True)
            if distributed:
                dist.barrier()
    finally:
        if lock:
            lock.close()
        if distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0, help="Only for a separate smoke cache; 0 builds all conditions")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit cannot be negative")
    build(resolve_config([args.config], args.set), args.limit)
