"""Composable YAML configs with strict overrides and no training-time implicit paths."""

import argparse
import copy
import json
import os
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import yaml

ROOT = Path(__file__).resolve().parents[1]


def merge(base, patch):
    result = copy.deepcopy(base)
    for k, v in patch.items():
        result[k] = (
            merge(result[k], v)
            if isinstance(v, dict) and isinstance(result.get(k), dict)
            else copy.deepcopy(v)
        )
    return result


def read_config(path, stack=()):
    path = Path(path).resolve()
    if path in stack:
        raise ValueError(f"Config inheritance cycle at {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a mapping")
    parents = value.pop("extends", [])
    if isinstance(parents, str):
        parents = [parents]
    base = {}
    for parent in parents:
        base = merge(base, read_config(path.parent / parent, stack + (path,)))
    return merge(base, value)


def override(cfg, value):
    key, sep, raw = value.partition("=")
    if not sep:
        raise ValueError("--set requires dotted.key=value")
    keys = key.split(".")
    node = cfg
    for k in keys[:-1]:
        if k not in node:
            raise ValueError(f"Unknown override {key}")
        node = node[k]
    if keys[-1] not in node and not any(k.endswith("kwargs") for k in keys):
        raise ValueError(f"Unknown override {key}")
    node[keys[-1]] = yaml.safe_load(raw)


def validate(cfg):
    m, d, t = cfg["model"], cfg["data"], cfg["training"]
    if d.get("cache", {}).get("mode", "off") not in ("off", "read", "replay"):
        raise ValueError("data.cache.mode must be off, read or replay")
    if m["family"] not in ("wan_vace", "wan_action", "wan_physcanvas"):
        raise ValueError("model.family must be wan_vace, wan_action or wan_physcanvas")
    if m["dtype"] not in ("bfloat16", "float32", "float16"):
        raise ValueError("Unsupported model.dtype")
    from .models.action import normalize_action_mode

    m["action_mode"] = normalize_action_mode(m["action_mode"])
    if m["vace_in_dim"] not in (48, 54, 72, 144):
        raise ValueError("VACE controls must have 48,54,72,144 channels")
    if d["context_frames"] not in (1, 5) or d["horizon_frames"] <= 0 or d["horizon_frames"] % 4:
        raise ValueError("Use context 1/5 and a positive horizon divisible by 4")
    for k in ("save_steps", "eval_steps", "log_steps", "num_epochs", "gradient_accumulation_steps"):
        if not isinstance(t[k], int) or t[k] <= 0:
            raise ValueError(f"training.{k} must be a positive integer")
    if t["max_steps"] < 0 or t["low_noise_offset"] < 0:
        raise ValueError("Step limits cannot be negative")
    allowed = {"dit", "vace", "action"}
    if not set(t["trainable"]) <= allowed or not t["trainable"]:
        raise ValueError("Invalid trainable component")
    if m["action_dim"] <= 0:
        raise ValueError("action_dim must be positive")
    if m["family"] == "wan_action" and "vace" in t["trainable"]:
        raise ValueError("wan_action cannot train a VACE branch")
    if m["family"] == "wan_vace" and "action" in t["trainable"]:
        raise ValueError("wan_vace cannot train direct action projections")
    if m["family"] == "wan_vace" and m["view_embeddings"] not in (0, 1, 3):
        raise ValueError("view_embeddings must be 0,1,3")
    if m["family"] == "wan_vace" and d["view_mode"] == "single" and m["view_embeddings"] > 1:
        raise ValueError("Single view requires at most one view embedding")
    if d["name"] not in ("robotwin", "robotwin_endpose"):
        raise ValueError("This release supports robotwin and robotwin_endpose datasets")
    if m["family"] == "wan_physcanvas":
        if t["trainable"] != ["dit"] or not m["use_text"]:
            raise ValueError("PhysCanvas trains DiT plus replay embedding and uses text context")
    if cfg.get("condition_cache", {}).get("allow_partial_for_smoke") and not 0 < t["max_steps"] <= 3:
        raise ValueError("Partial condition cache is restricted to <=3-step smoke tests")
    return cfg


def resolve_config(paths, overrides=()):
    cfg = {}
    for path in paths:
        cfg = merge(cfg, read_config(path))
    for item in overrides:
        override(cfg, item)
    validate(cfg)
    def expand(value):
        if isinstance(value, list):
            return [expand(v) for v in value]
        if not value:
            return value
        value = str(value).replace("${WOVR_ASSET_ROOT}", os.environ.get("WOVR_ASSET_ROOT", str(ROOT.parent / "wovr-assets")))
        value = os.path.expandvars(os.path.expanduser(value))
        if "$" in value:
            raise ValueError(f"Unresolved environment variable in path: {value}")
        return str(Path(value).resolve() if Path(value).is_absolute() else (ROOT / value).resolve())

    for section, key in [
        ("model", "model_dir"), ("model", "tokenizer_path"), ("model", "checkpoint"),
        ("data", "base_path"), ("data", "train_path"), ("data", "val_path"),
        ("data", "fk_config"), ("data", "action_stats"), ("output", "root"),
    ]:
        cfg[section][key] = expand(cfg[section][key])
    cfg["data"]["cache"]["root"] = expand(cfg["data"]["cache"]["root"])
    if cfg.get("condition_cache"):
        cfg["condition_cache"]["root"] = expand(cfg["condition_cache"]["root"])
    return cfg


def run_path(cfg):
    o, m, d, t = cfg["output"], cfg["model"], cfg["data"], cfg["training"]
    stamp = os.environ.get("WOVR_RUN_TIMESTAMP") or datetime.now(
        ZoneInfo("Asia/Shanghai")
    ).strftime("%Y%m%d-%H%M%S")
    task = (
        o["task"]
        or f"{d['name']}-{m['family']}-{m['action_mode'] if m['family']=='wan_action' else 'vace'+str(m['vace_in_dim'])}-{d['view_mode']}-{d['sampling']['sample_mode']}-blur{d['sampling']['context_sphere_blur_prob']}-c{d['context_frames']}h{d['horizon_frames']}-{'_'.join(t['trainable'])}-lr{t['learning_rate']:g}"
    )
    task = re.sub(r"[^a-zA-Z0-9_.-]+", "_", task)
    if d.get("cache", {}).get("mode", "off") != "off":
        task += "-cache-" + d["cache"]["mode"]
    return Path(o["root"]) / (stamp + "-" + task)


def parser():
    p = argparse.ArgumentParser(description="WoVR+ WAN world models")
    p.add_argument(
        "--config", action="append", default=None, help="YAML file; repeat to merge overlays"
    )
    p.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", help="Override a dotted YAML key"
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved configuration without opening models/data",
    )
    p.add_argument(
        "--validate-only",
        action="store_true",
        help="Check model/data files and one train/val sample without model allocation",
    )
    return p
