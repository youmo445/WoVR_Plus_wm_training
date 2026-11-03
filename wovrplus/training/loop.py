"""Step-based action-conditioned world-model training and distributed evaluation."""

import json
import os
import time
import sys
import hashlib
import tarfile
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from .profiling import StepProfiler
from .distributed import reject_optimizer_offload
from .loss_metrics import LossMetrics


def save_run_config(args):
    if (
        min(
            args.save_steps,
            args.eval_steps,
            args.log_steps,
            args.num_epochs,
            args.gradient_accumulation_steps,
        )
        <= 0
        or args.max_steps < 0
    ):
        raise ValueError("Training intervals must be positive; max_steps must be nonnegative")
    set_seed(args.seed)
    if int(os.environ.get("RANK", "0")) != 0:
        return
    root = Path(args.output_path)
    root.mkdir(parents=True, exist_ok=True)
    (root / "checkpoints").mkdir(exist_ok=True)
    from wovrplus.config import ROOT

    cfg = getattr(args, "resolved_config", {})
    family = cfg.get("model", {}).get("family", "raymap_vace")
    sources = sorted(
        p
        for pattern in (
            "wovrplus/**/*.py",
            "configs/**/*.yaml",
            "configs/**/*.json",
            "scripts/*.sh",
            "pyproject.toml",
        )
        for p in ROOT.glob(pattern)
    )
    config = {
        "config": cfg,
        "arguments": vars(args),
        "command": sys.argv,
        "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "initialization": (
            "trained_checkpoint"
            if args.init_checkpoint_path
            else (
                "original_wan_dit_with_vace_from_dit"
                if family == "raymap_vace" and not cfg.get("model", {}).get("vace_random_init", False)
                else "original_wan_dit_with_new_condition_projections"
            )
        ),
        "cwd": os.getcwd(),
        "torch_version": torch.__version__,
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
    }
    revision = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    if revision.returncode == 0:
        config["git_commit"] = revision.stdout.strip()
        config["git_dirty"] = bool(
            subprocess.run(
                ["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True
            ).stdout.strip()
        )
    # Refuse to silently reuse/overwrite an earlier experiment.
    with (root / "config.json").open("x", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    with tarfile.open(root / "source_snapshot.tar.gz", "w:gz") as tar:
        for p in sources:
            tar.add(p, arcname=str(p.relative_to(ROOT)), recursive=False)


def evaluate(model, loader, accelerator, max_steps, round_id):
    # A fresh permutation each evaluation, synchronized across all ranks.
    loader.set_epoch(round_id)
    if getattr(loader, "synchronized_generator", None) is not None:
        loader.synchronized_generator.manual_seed(1729 + round_id)
    module = accelerator.unwrap_model(model)
    if hasattr(module, "begin_evaluation"):
        module.begin_evaluation(round_id)
    model.eval()
    total = torch.zeros(2, device=accelerator.device, dtype=torch.float64)
    metrics = LossMetrics(accelerator.device)
    try:
        with torch.no_grad():
            for index, data in enumerate(loader):
                loss = (
                    model({}, inputs=data)
                    if getattr(loader.dataset, "load_from_cache", False)
                    else model(data)
                )
                total[0] += loss.detach().double()
                total[1] += 1
                metrics.add(loss, getattr(module, "last_loss_metrics", None))
                if index + 1 >= max_steps:
                    break
        total = accelerator.reduce(total, reduction="sum")
        diagnostic = LossMetrics.summarize(accelerator.reduce(metrics.sums, reduction="sum"))
        return (total[0] / total[1]).item(), int(total[1].item()), diagnostic
    finally:
        model.train()


def create_accelerator(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)
        ],
    )
    plugin = accelerator.state.deepspeed_plugin
    if plugin is not None:
        reject_optimizer_offload(plugin.deepspeed_config)
    return accelerator


def launch_training_task(dataset, val_dataset, model, model_logger, args, accelerator=None):
    if args.save_steps <= 0 or args.eval_steps <= 0:
        raise ValueError("save_steps and eval_steps must be positive")
    accelerator = accelerator or create_accelerator(args)
    set_seed(args.seed)
    training_module = model
    rng_audit = None
    if hasattr(model, "configure_diffusion_rng"):
        streams = model.configure_diffusion_rng(args.seed, accelerator.process_index, accelerator.device)
        print("DIFFUSION_RNG_CONFIG " + json.dumps(streams), flush=True)
        seed_row = torch.tensor([[accelerator.process_index,
                                  streams["train"]["noise_seed"], streams["train"]["timestep_seed"],
                                  streams["validation"]["noise_seed"], streams["validation"]["timestep_seed"]]],
                                dtype=torch.int64, device=accelerator.device)
        seed_rows = accelerator.gather(seed_row).cpu().tolist()
        assert len({row[1] for row in seed_rows}) == accelerator.num_processes
        assert len({row[2] for row in seed_rows}) == accelerator.num_processes
        rng_audit = dict(scheme="rank-phase-domain-v1", data_seed=args.seed,
                         train_validation_isolated=True,
                         columns=["rank", "train_noise_seed", "train_timestep_seed",
                                  "initial_validation_noise_seed", "initial_validation_timestep_seed"],
                         per_rank_seeds=seed_rows)
    weights = getattr(dataset, "sample_weights", None)
    if weights is not None:
        if len(weights) != len(dataset):
            raise ValueError("Weighted dataset requires one weight per sample")
        sampler = torch.utils.data.WeightedRandomSampler(
            weights,
            int(getattr(dataset, "weighted_num_samples", len(dataset))),
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
        train_loader = DataLoader(
            dataset,
            sampler=sampler,
            collate_fn=lambda x: x[0],
            num_workers=args.dataset_num_workers,
        )
    else:
        train_loader = DataLoader(
            dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=args.dataset_num_workers
        )
    val_loader = DataLoader(
        val_dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=args.dataset_num_workers
    )
    optimizer = torch.optim.AdamW(
        model.trainable_modules(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    # Preserve the original scheduler (initial ConstantLR warm-up).
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    if getattr(getattr(model, "optimizer", None), "cpu_offload", False):
        raise RuntimeError("Optimizer CPU offload is forbidden in this project")
    if accelerator.gradient_accumulation_steps != args.gradient_accumulation_steps:
        raise ValueError(
            "Accelerate/DeepSpeed gradient accumulation differs from training config; use the supplied auto-valued zero2.json"
        )
    output = Path(args.output_path)
    writer = SummaryWriter(str(output / "tensorboard")) if accelerator.is_main_process else None
    profiler = StepProfiler(accelerator, writer, output, getattr(args, "profiling", {}))
    profiler.attach(model)
    if writer:
        config = json.loads((output / "config.json").read_text())
        config["runtime"] = {
            "world_size": accelerator.num_processes,
            "global_batch_size": accelerator.num_processes * args.gradient_accumulation_steps,
            "train_samples": len(dataset),
            "validation_samples": len(val_dataset),
            "batches_per_full_epoch": len(train_loader),
            "checkpoint_format": "complete DiT and optional VACE weights; frozen VAE/text encoder and optimizer state excluded",
            "deepspeed": getattr(accelerator, "deepspeed_config", None),
            "trainable_parameters": sum(
                p.numel() for p in accelerator.unwrap_model(model).parameters() if p.requires_grad
            ),
            "latent_cache": {
                split: {
                    k: getattr(ds, "manifest", {}).get(k)
                    for k in ("signature", "variants", "selected_windows", "raw_windows")
                }
                for split, ds in (("train", dataset), ("val", val_dataset))
            },
            "diffusion_rng": rng_audit,
            "loss_logging": dict(
                train_step="Current optimizer-update mean across ranks and accumulation microbatches",
                train_interval="All microbatch losses since preceding log boundary, reduced across all ranks",
                unweighted_mse="Same horizon mask as training objective, before timestep weighting",
                timestep_bins="10 bins of actual scheduler timestep / num_train_timesteps; last includes 1.0",
            ),
        }
        (output / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
    global_step = 0
    pending_loss = torch.zeros(2, device=accelerator.device)
    interval_metrics = LossMetrics(accelerator.device)
    last_logged_step = 0

    def log_interval(step):
        nonlocal last_logged_step
        report = LossMetrics.summarize(accelerator.reduce(interval_metrics.sums, reduction="sum"))
        report.update(phase="train", step=step, interval_start_step=last_logged_step + 1,
                      optimizer_steps=step - last_logged_step)
        if writer:
            LossMetrics.write(writer, "train", step, report)
            writer.add_scalar("Counts/train_interval_optimizer_steps", report["optimizer_steps"], step)
            with (output / "loss_metrics.jsonl").open("a") as f:
                f.write(json.dumps(report) + "\n")
            print(f"TRAIN_INTERVAL_COMPLETE step={step} start={report['interval_start_step']} "
                  f"loss={report['weighted_loss']:.6f} unweighted_mse={report['unweighted_mse']:.6f} "
                  f"samples={report['sample_count']}", flush=True)
        interval_metrics.reset()
        last_logged_step = step
    model.train()
    optimizer.zero_grad()
    try:
        for epoch in range(args.num_epochs):
            train_loader.set_epoch(epoch)
            epoch_loss = torch.zeros(2, device=accelerator.device)
            fetch_start = time.perf_counter()
            for data in train_loader:
                profiler.begin(fetch_start, time.perf_counter() - fetch_start, global_step + 1)
                with accelerator.accumulate(model):
                    loss = (
                        model({}, inputs=data)
                        if getattr(dataset, "load_from_cache", False)
                        else model(data)
                    )
                    if profiler.engine_wrapped:
                        accelerator.backward(loss)
                        optimizer.step()
                    else:
                        with profiler.section("backward_and_gradient_communication"):
                            accelerator.backward(loss)
                        with profiler.section("optimizer_and_parameter_communication"):
                            optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    pending_loss[0] += loss.detach().float()
                    pending_loss[1] += 1
                    interval_metrics.add(loss, getattr(training_module, "last_loss_metrics", None))
                if not accelerator.sync_gradients:
                    fetch_start = time.perf_counter()
                    continue
                global_step += 1
                epoch_loss += pending_loss
                if (global_step <= 10 or global_step % args.log_steps == 0
                        or (args.max_steps and global_step == args.max_steps)):
                    reduced = accelerator.reduce(pending_loss, reduction="sum")
                    value = (reduced[0] / reduced[1]).item()
                    log_interval(global_step)
                    if accelerator.device.type == "cuda":
                        torch.cuda.synchronize(accelerator.device)
                    if writer:
                        writer.add_scalar("Loss/train_step", value, global_step)
                        writer.add_scalar(
                            "LearningRate", optimizer.param_groups[0]["lr"], global_step
                        )
                        writer.flush()
                        print(
                            f"TRAIN_UPDATE_COMPLETE step={global_step} loss={value:.6f} unix_time={time.time():.3f}",
                            flush=True,
                        )
                pending_loss.zero_()
                profiler.finish(global_step)
                if global_step % args.save_steps == 0:
                    print(
                        f"CHECKPOINT_START step={global_step} rank={accelerator.process_index}",
                        flush=True,
                    )
                    model_logger.save_model(
                        accelerator, model, f"step-{global_step:08d}.safetensors"
                    )
                    accelerator.wait_for_everyone()
                    value, count, validation_metrics = evaluate(
                        model,
                        val_loader,
                        accelerator,
                        args.eval_steps,
                        global_step // args.save_steps,
                    )
                    if writer:
                        writer.add_scalar("Loss/val_step", value, global_step)
                        LossMetrics.write(writer, "validation", global_step, validation_metrics)
                        with (output / "loss_metrics.jsonl").open("a") as f:
                            f.write(json.dumps(dict(validation_metrics, phase="validation", step=global_step)) + "\n")
                        writer.flush()
                        print(
                            f"CHECKPOINT_AND_EVAL_COMPLETE step={global_step} eval_batches={count // accelerator.num_processes} loss={value:.6f}",
                            flush=True,
                        )
                if args.max_steps and global_step >= args.max_steps:
                    break
                fetch_start = time.perf_counter()
            reduced = accelerator.reduce(epoch_loss, reduction="sum")
            if writer and reduced[1] > 0:
                writer.add_scalar("Loss/epoch", (reduced[0] / reduced[1]).item(), epoch)
                writer.flush()
            if args.max_steps and global_step >= args.max_steps:
                break
        if global_step > last_logged_step:
            log_interval(global_step)
        if (
            global_step
            and global_step % args.save_steps
            and getattr(args, "save_final_checkpoint", True)
        ):
            model_logger.save_model(accelerator, model, f"step-{global_step:08d}-final.safetensors")
    finally:
        profiler.close()
        if writer:
            writer.close()
