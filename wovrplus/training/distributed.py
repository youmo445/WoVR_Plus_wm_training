"""Launch policy: optimizer states must stay on the accelerator."""

import argparse
import json
from pathlib import Path
import yaml


def reject_optimizer_offload(config):
    zero = config.get("zero_optimization", {})
    offload = zero.get("offload_optimizer", {}) or {}
    device = offload.get("device", "none") if isinstance(offload, dict) else offload
    if str(device).lower() not in ("none", "null"):
        raise ValueError(
            "Optimizer CPU/NVMe offload is disabled by project policy; use device=none"
        )


def check_launch_config(path):
    config = yaml.safe_load(Path(path).read_text())
    ds = config.get("deepspeed_config", {}) or {}
    device = ds.get("offload_optimizer_device", "none")
    reject_optimizer_offload({"zero_optimization": {"offload_optimizer": {"device": device}}})
    if ds.get("deepspeed_config_file"):
        # Accelerate resolves this filename against the launch working directory.
        reject_optimizer_offload(json.loads(Path(ds["deepspeed_config_file"]).read_text()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    check_launch_config(parser.parse_args().config)
