"""Compare real online/cached inputs and loss for all variants across all five tasks."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import torch
from wovrplus.config import resolve_config
from wovrplus.data.registry import build_dataset
from wovrplus.pipelines.world_model import WorldModelPipeline


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    cfg = resolve_config([args.config])
    cfg["data"]["cache"]["mode"] = "read"
    cfg["data"]["cache"]["verify_checksum"] = True
    pipe = WorldModelPipeline.from_config(cfg).to("cuda").eval().requires_grad_(False)
    pipe.scheduler.set_timesteps(1000, training=True)
    report = []
    for split in ("train", "val"):
        cached = build_dataset(cfg, split)
        online_cfg = deepcopy(cfg)
        online_cfg["data"]["cache"]["mode"] = "replay"
        online = build_dataset(online_cfg, split)
        selected = {}
        for i, r in enumerate(cached.records):
            selected.setdefault((r["meta"]["task"], r["variant"]), i)
        for (task, variant), i in selected.items():
            sample = online[i]
            stored = cached[i]
            assert sample["meta"] == stored["meta"], (task, variant, "metadata")
            torch.manual_seed(42 + i)
            a = pipe.prepare_sample(sample)
            torch.manual_seed(42 + i)
            b = pipe.prepare_sample(stored)
            errors = {}
            for key in a:
                if torch.is_tensor(a[key]):
                    errors[key] = float((a[key].float() - b[key].float()).abs().max())
                    torch.testing.assert_close(a[key], b[key], rtol=0, atol=0, msg=key)
            kwargs = dict(
                dit=pipe.dit,
                vace=pipe.vace,
                use_gradient_checkpointing=False,
                clean_latent_slots=2,
                ti2v2_low_noise_step_offset=1,
                ti2v2_horizon_loss_latent_slots=2,
            )
            with torch.no_grad():
                torch.manual_seed(314 + i)
                loss_a = pipe.training_loss(**kwargs, **a)
                torch.manual_seed(314 + i)
                loss_b = pipe.training_loss(**kwargs, **b)
            torch.testing.assert_close(loss_a, loss_b, rtol=1e-5, atol=1e-6)
            report.append(
                dict(
                    split=split,
                    task=task,
                    variant=variant,
                    max_abs_errors=errors,
                    online_loss=float(loss_a),
                    cached_loss=float(loss_b),
                )
            )
            print(
                f"CACHE_NUMERICAL_PASS split={split} task={task} variant={variant} loss={float(loss_b):.6f}",
                flush=True,
            )
    Path(args.output).write_text(json.dumps({"samples": len(report), "checks": report}, indent=2))


if __name__ == "__main__":
    main()
