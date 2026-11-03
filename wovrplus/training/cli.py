import argparse
import contextlib
import json
import os
import sys
from pathlib import Path
from wovrplus.config import ROOT, parser, resolve_config, run_path


def preflight(cfg):
    from wovrplus.data.condition_cache import build_dataset
    from safetensors import safe_open
    import torch

    root = Path(cfg["model"]["model_dir"])
    shards = sorted(root.glob("diffusion_pytorch_model-*-of-*.safetensors")) or [
        root / "diffusion_pytorch_model.safetensors"
    ]
    paths = shards + [root / "Wan2.2_VAE.pth"]
    if cfg["model"]["family"] == "raymap_vace" or cfg["model"]["use_text"]:
        paths += [root / "models_t5_umt5-xxl-enc-bf16.pth", Path(cfg["model"]["tokenizer_path"])]
    if cfg["model"]["checkpoint"]:
        paths.append(Path(cfg["model"]["checkpoint"]))
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    shapes = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in shapes:
                    raise ValueError(f"Duplicate base parameter {key}")
                shapes[key] = tuple(f.get_slice(key).get_shape())
    from wovrplus.models.loading import WAN_5B
    from wovrplus.models.wan_dit import WanModel
    from wovrplus.models.utils import init_weights_on_device

    with init_weights_on_device():
        expected = WanModel(**WAN_5B)
    expected_shapes = {k: tuple(v.shape) for k, v in expected.state_dict().items()}
    if shapes != expected_shapes:
        raise ValueError("Base weight keys/shapes do not match WAN 2.2 TI2V 5B")
    del expected
    datasets = [build_dataset(cfg, split) for split in ("train", "val")]
    samples = []
    for ds in datasets:
        s = ds[0]
        if "latent_cache" in s:
            samples.append(
                {
                    "cached_tensors": {k: list(v.shape) for k, v in s["latent_cache"].items()},
                    "keys": list(s),
                }
            )
            continue
        if len(s["video"]) != cfg["data"]["context_frames"] + cfg["data"]["horizon_frames"]:
            raise ValueError("Dataset frame count does not match config")
        samples.append(
            {
                "frames": len(s["video"]),
                "image_size": s["video"][0].size,
                "action_shape": list(s["action"].shape) if "action" in s else None,
                "keys": list(s),
            }
        )
    return {
        "base_tensor_keys": len(shapes),
        "base_shapes_match": True,
        "train_samples": len(datasets[0]),
        "val_samples": len(datasets[1]),
        "samples": samples,
    }


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, value):
        self.stream.write(value)
        self.log.write(value)
        self.log.flush()

    def flush(self):
        self.stream.flush()
        self.log.flush()

    def isatty(self):
        return False


def main():
    cli = parser().parse_args()
    cfg = resolve_config(cli.config or [ROOT / "configs/train/robotwin_raymap_vace.yaml"], cli.set)
    if cli.dry_run:
        print(json.dumps(cfg, ensure_ascii=False, indent=2))
        return
    os.chdir(ROOT)
    if cli.validate_only:
        print("PREFLIGHT_PASS " + json.dumps(preflight(cfg), ensure_ascii=False))
        return
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not os.environ.get("WOVR_RUN_TIMESTAMP"):
        raise ValueError(
            "Distributed launches must use scripts/train.sh or set a shared WOVR_RUN_TIMESTAMP"
        )
    output = run_path(cfg)
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    args = argparse.Namespace(
        **cfg["training"],
        output_path=str(output),
        init_checkpoint_path=cfg["model"]["checkpoint"],
        resolved_config=cfg,
        profiling=cfg.get("profiling", {}),
    )
    from .loop import save_run_config, launch_training_task, create_accelerator
    from .module import WorldModelTrainingModule, CheckpointWriter
    from wovrplus.data.condition_cache import build_dataset

    rank = int(os.environ.get("RANK", "0"))
    with (output / "logs" / f"rank-{rank}.log").open("a", buffering=1) as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(
            Tee(sys.stderr, log)
        ):
            # Validate the distributed configuration and select the local device before
            # loading large checkpoints. Invalid launch configs should fail cheaply.
            accelerator = create_accelerator(args)
            save_run_config(args)
            train, val = build_dataset(cfg, "train"), build_dataset(cfg, "val")
            model = WorldModelTrainingModule(cfg)
            launch_training_task(
                train, val, model, CheckpointWriter(str(output / "checkpoints")), args, accelerator
            )


if __name__ == "__main__":
    main()
