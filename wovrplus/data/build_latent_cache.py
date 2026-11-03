"""Offline RoboTwin window cache; torchrun shards work over available devices."""

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
import yaml
import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import save_file
from wovrplus.config import ROOT, resolve_config
from wovrplus.models.loading import load_vae
from wovrplus.pipelines.world_model import WorldModelPipeline
from .registry import build_dataset
from .latent_cache import (
    SCHEMA,
    code_fingerprint,
    digest_file,
    encoding_spec,
    file_identity,
    replay_sample,
    variant_seed,
)


def atomic_json(path, value):
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def make_plan(cfg, datasets, variants, windows_per_episode, seed):
    vae_path = Path(cfg["model"]["model_dir"]) / "Wan2.2_VAE.pth"
    common = {
        "schema": SCHEMA,
        "encoding_spec": encoding_spec(cfg),
        "code_sha256": code_fingerprint(),
        "vae_identity": file_identity(vae_path),
        "vae_sha256": digest_file(vae_path),
        "variants": variants,
        "windows_per_episode": windows_per_episode,
        "seed": seed,
    }
    assets = []
    fk = Path(cfg["data"]["fk_config"])
    assets.append(file_identity(fk))
    urdf = Path(yaml.safe_load(fk.read_text())["urdf_path"])
    assets.append(file_identity(urdf if urdf.is_absolute() else fk.parent / urdf))
    if cfg["data"]["action_stats"]:
        assets.append(file_identity(cfg["data"]["action_stats"]))
    common["assets"] = assets
    splits = {}
    for split, adapter in datasets.items():
        dataset = adapter.dataset
        indices = defaultdict(list)
        for index, (episode, _) in enumerate(dataset.sample_indices):
            indices[episode].append(index)
        selected = []
        for entries in indices.values():
            if windows_per_episode == 0:
                selected.extend(entries)
            else:
                count = min(windows_per_episode, len(entries))
                selected.extend(
                    entries[min(len(entries) - 1, int((i + 0.5) * len(entries) / count))]
                    for i in range(count)
                )
        records = [
            dict(
                source_index=index,
                variant=v,
                seed=variant_seed(seed, split, index, v),
                file=f"tensors/{position*variants+v:09d}.safetensors",
            )
            for position, index in enumerate(selected)
            for v in range(variants)
        ]
        splits[split] = {
            "raw_windows": len(dataset),
            "selected_windows": len(selected),
            "sources": [file_identity(ep["path"]) for ep in dataset.episode_info],
            "records": records,
        }
    signature = hashlib.sha256(
        json.dumps({"common": common, "splits": splits}, sort_keys=True).encode()
    ).hexdigest()
    return {"common": common, "splits": splits, "signature": signature}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--variants", type=int, default=4)
    parser.add_argument(
        "--windows-per-episode",
        type=int,
        default=0,
        help="0=all windows; positive is an explicit validation subset",
    )
    parser.add_argument("--seed", type=int, default=20261029)
    args = parser.parse_args()
    if args.variants <= 0 or args.windows_per_episode < 0:
        parser.error("variants must be positive and windows-per-episode nonnegative")
    cfg = resolve_config(args.config, args.set)
    if cfg["data"]["name"] != "robotwin":
        raise ValueError("Offline cache builder currently supports the RoboTwin adapter only")
    cfg["data"]["cache"]["mode"] = "off"
    root = Path(cfg["data"]["cache"]["root"]).resolve()
    if root.is_relative_to(ROOT.resolve()):
        raise ValueError("Store latent caches outside the source checkout")
    rank, world, local = (
        int(os.environ.get(k, d))
        for k, d in (("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0"))
    )
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("nccl")
    start = time.perf_counter()
    datasets = {split: build_dataset(cfg, split) for split in ("train", "val")}
    root.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        plan = make_plan(cfg, datasets, args.variants, args.windows_per_episode, args.seed)
        plan_path = root / "plan.json"
        if (
            plan_path.exists()
            and json.loads(plan_path.read_text())["signature"] != plan["signature"]
        ):
            raise ValueError(
                "Existing cache has a different plan; choose a new external cache directory"
            )
        atomic_json(plan_path, plan)
        print(
            "CACHE_PLAN "
            + json.dumps(
                {
                    s: {k: v for k, v in p.items() if k not in ("records", "sources")}
                    for s, p in plan["splits"].items()
                }
            ),
            flush=True,
        )
    if world > 1:
        dist.barrier()
    plan = json.loads((root / "plan.json").read_text())
    dtype = getattr(torch, cfg["model"]["dtype"])
    vae = load_vae(cfg["model"]["model_dir"], dtype)
    control = None
    if cfg["model"]["family"] == "wan_vace":
        control = torch.nn.Module()
        control.vace_in_dim = cfg["model"]["vace_in_dim"]
    pipe = WorldModelPipeline(None, vae, vace=control, dtype=dtype).to(f"cuda:{local}").eval()
    count, resumed = 0, 0
    for split, split_plan in plan["splits"].items():
        directory = root / split
        (directory / "tensors").mkdir(parents=True, exist_ok=True)
        records = []
        for i in range(rank, len(split_plan["records"]), world):
            record = deepcopy(split_plan["records"][i])
            path = directory / record["file"]
            reusable = False
            if path.exists():
                try:
                    with safe_open(str(path), framework="pt", device="cpu") as f:
                        metadata = f.metadata()
                        reusable = metadata.get("signature") == plan["signature"]
                        if reusable:
                            record.update(json.loads(metadata["record"]))
                            for key in f.keys():
                                f.get_tensor(
                                    key
                                )  # reject truncated files before publishing a manifest
                except Exception:
                    reusable = False
            if reusable:
                resumed += 1
            else:
                sample = replay_sample(datasets[split], record)
                inputs = pipe.prepare_sample(sample)
                keys = (
                    ("input_latents", "vace_context")
                    if control is not None
                    else ("input_latents", "action")
                )
                tensors = {k: inputs[k].detach().cpu().contiguous() for k in keys}
                record.update(
                    prompt=sample.get("prompt", ""),
                    meta=sample["meta"],
                    tensors={
                        k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                        for k, v in tensors.items()
                    },
                )
                temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
                save_file(
                    tensors,
                    str(temp),
                    metadata={
                        "signature": plan["signature"],
                        "record": json.dumps(record, ensure_ascii=False),
                    },
                )
                os.replace(temp, path)
            record["sha256"] = digest_file(path)
            records.append(record)
            count += 1
            if count % 25 == 0:
                print(
                    f"CACHE_PROGRESS rank={rank} samples={count} elapsed={time.perf_counter()-start:.1f}s",
                    flush=True,
                )
        atomic_json(directory / f"records-rank{rank}.json", records)
    if world > 1:
        dist.barrier()
    if rank == 0:
        for split, split_plan in plan["splits"].items():
            records = []
            for r in range(world):
                records.extend(json.loads((root / split / f"records-rank{r}.json").read_text()))
            records.sort(key=lambda r: r["file"])
            if len(records) != len(split_plan["records"]) or len(
                {r["file"] for r in records}
            ) != len(records):
                raise ValueError("Incomplete distributed cache build")
            manifest = {
                **plan["common"],
                **split_plan,
                "signature": plan["signature"],
                "complete": True,
                "records": records,
            }
            atomic_json(root / split / "manifest.json", manifest)
        atomic_json(
            root / "build_summary.json",
            dict(
                elapsed_s=time.perf_counter() - start,
                world_size=world,
                samples={s: len(v["records"]) for s, v in plan["splits"].items()},
                signature=plan["signature"],
            ),
        )
        print("CACHE_BUILD_COMPLETE " + str(root), flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
