"""Generate a held-out dataset chunk using the same model/config as training."""

import json
import os
from pathlib import Path
import torch
from wovrplus.config import ROOT, parser, resolve_config


def main():
    p = parser()
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cfg-scale", type=float, default=1.0)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    cfg = resolve_config(args.config or [ROOT / "configs/train/robotwin_vace.yaml"], args.set)
    if args.dry_run:
        print(json.dumps(cfg, indent=2))
        return
    os.chdir(ROOT)
    if args.validate_only:
        from wovrplus.training.cli import preflight

        print(json.dumps(preflight(cfg), indent=2))
        return
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    from wovrplus.data.registry import build_dataset
    from wovrplus.pipelines.world_model import WorldModelPipeline

    cfg["data"]["cache"]["mode"] = "off"
    cfg.pop("condition_cache", None)
    dataset = build_dataset(cfg, "val")
    sample = dataset[args.index]
    pipeline = WorldModelPipeline
    if cfg["model"]["family"] == "wan_physcanvas":
        from wovrplus.pipelines.physcanvas import PhysCanvasPipeline
        pipeline = PhysCanvasPipeline
    pipe = pipeline.from_config(cfg).to(args.device).eval()
    video = pipe.generate(
        sample,
        context_frames=cfg["data"]["context_frames"],
        steps=args.steps,
        seed=args.seed,
        cfg_scale=args.cfg_scale,
    )
    import imageio.v3 as iio
    import numpy as np

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output, np.stack([np.asarray(frame) for frame in video]), fps=30)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "config": cfg,
                "sample": sample.get("meta", {}),
                "steps": args.steps,
                "seed": args.seed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
