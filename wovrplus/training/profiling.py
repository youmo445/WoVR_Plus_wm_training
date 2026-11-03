"""Opt-in stage timing, per-rank memory and TensorBoard distributions.

CUDA events measure stream spans (including waits), not pure kernel active time.
The synchronized step wall timer is the primary throughput measurement.
"""

from contextlib import contextmanager, nullcontext
from pathlib import Path
import json
import time
import numpy as np
import torch
import torch.distributed as dist


class StepProfiler:
    def __init__(self, accelerator, writer, output, options):
        self.accelerator = accelerator
        self.writer = writer
        self.output = Path(output)
        self.options = options
        self.enabled = bool(options.get("enabled", False))
        self.cuda = accelerator.device.type == "cuda"
        self.active = False
        self.events = []
        self.host = {}
        self.counts = {}
        self.details = {}
        self.records = []
        self.last_end = None
        self.engine_wrapped = False
        self.trace = None
        self.trace_step = int(options.get("trace_step", 0))

    @contextmanager
    def section(self, name):
        if not self.active:
            yield
            return
        start = end = None
        if self.cuda:
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
        wall = time.perf_counter()
        self.counts[name] = self.counts.get(name, 0) + 1
        with torch.profiler.record_function("wovr/" + name):
            yield
        self.host[name] = self.host.get(name, 0.0) + time.perf_counter() - wall
        if self.cuda:
            end.record()
            self.events.append((name, start, end))

    def _wrap(self, obj, attr, name):
        original = getattr(obj, attr)

        def wrapped(*args, **kwargs):
            label = name() if callable(name) else name
            with self.section(label):
                return original(*args, **kwargs)

        setattr(obj, attr, wrapped)

    def attach(self, model):
        if not self.enabled:
            return
        module = self.accelerator.unwrap_model(model)
        pipe = getattr(module, "pipe", None)
        if pipe is not None and hasattr(pipe, "prepare_sample"):
            detail = self.options.get("encoder_detail", True)
            if detail:
                pipe._profile_section = self.section
            original_prepare = pipe.prepare_sample

            def prepare(*args, **kwargs):
                self.encode_count = 0
                self.details["text_encoder_present"] = pipe.text_encoder is not None
                self.details["prompt_nonempty"] = bool(args[0].get("prompt", ""))
                self.details["latent_cache_hit"] = "latent_cache" in args[0]
                with self.section("prepare_sample"):
                    return original_prepare(*args, **kwargs)

            pipe.prepare_sample = prepare

            original_encode = pipe.vae.encode

            def encode(*args, **kwargs):
                self.vae_label = "video_vae" if self.encode_count == 0 else "control_vae"
                self.chunk_index = 0
                self.encode_count += 1
                self.details[self.vae_label + "_input_shape"] = list(args[0].shape)
                with self.section(self.vae_label):
                    return original_encode(*args, **kwargs)

            def vae_detail(name):
                return "VAE/" + self.vae_label + "/" + name

            pipe.vae.encode = encode
            if detail:
                pipe.vae._profile_section = lambda name: self.section(vae_detail(name))
            if detail and hasattr(pipe.vae, "model"):
                self._wrap(pipe.vae.model, "encode", lambda: vae_detail("model_encode"))

                def chunk_label():
                    index = self.chunk_index
                    self.chunk_index += 1
                    return vae_detail(f"encoder_chunk_{index}")

                self._wrap(pipe.vae.model.encoder, "forward", chunk_label)
                self._wrap(
                    pipe.vae.model.conv1, "forward", lambda: vae_detail("moments_projection")
                )

            if pipe.text_encoder is not None:
                if detail:
                    self._wrap(pipe, "encode_text", "Text/total")
                original_text_forward = pipe.text_encoder.forward

                def text_forward(ids, mask, *args, **kwargs):
                    self.details["text_token_shape"] = list(ids.shape)
                    with self.section("text_encoder"):
                        return original_text_forward(ids, mask, *args, **kwargs)

                pipe.text_encoder.forward = text_forward
            self._wrap(pipe, "model_fn", "denoiser_forward")
            if pipe.vace is not None:
                self._wrap(pipe.vace, "forward", "vace_forward")
        # Accelerate's DeepSpeed wrapper runs engine.step *inside backward*.
        # Measure the engine methods instead of timing the no-op optimizer wrapper.
        if self.accelerator.distributed_type.value == "DEEPSPEED":
            self.engine = model
            masters = getattr(model.optimizer, "single_partition_of_fp32_groups", [])
            if not masters:
                masters = getattr(model.optimizer, "fp32_groups_flat", [])
            self.weight_probe = masters[0].detach().flatten()[:4096] if masters else None
            self.weight_probe_initial = (
                self.weight_probe.cpu().clone() if self.weight_probe is not None else None
            )
            self._wrap(model, "backward", "backward_and_gradient_communication")
            self._wrap(model, "step", "optimizer_and_parameter_communication")
            self.engine_wrapped = True

    def begin(self, data_start, data_wait, update_number):
        if not self.enabled:
            return
        if not self.active:
            self.start = data_start
            self.host = {"data_wait": data_wait}
            self.counts = {}
            self.details = {}
            self.events = []
            self.active = True
            if self.cuda:
                torch.cuda.reset_peak_memory_stats(self.accelerator.device)
            if update_number == self.trace_step and self.accelerator.is_main_process:
                activities = [torch.profiler.ProfilerActivity.CPU]
                if self.cuda:
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
                self.trace = torch.profiler.profile(activities=activities, record_shapes=False)
                self.trace.__enter__()
        else:
            self.host["data_wait"] += data_wait

    def finish(self, step):
        if not self.enabled:
            return
        if self.cuda:
            torch.cuda.synchronize(self.accelerator.device)
        end = time.perf_counter()
        stream = {}
        for name, begin, finish in self.events:
            stream[name] = stream.get(name, 0.0) + begin.elapsed_time(finish) / 1000.0
        if "vace_forward" in stream:
            stream["dit_forward_excluding_vace"] = max(
                0.0, stream["denoiser_forward"] - stream["vace_forward"]
            )
        record = {
            "rank": self.accelerator.process_index,
            "step": step,
            "update_wall_s": end - self.start,
            "observed_interval_s": end - self.last_end if self.last_end else end - self.start,
            "host_call_s": self.host,
            "gpu_stream_span_s": stream,
            "call_counts": dict(self.counts),
            "input_details": dict(self.details),
        }
        for name in ("video_vae", "control_vae", "text_encoder"):
            record["call_counts"].setdefault(name, 0)
        if self.engine_wrapped:
            record["engine_global_steps"] = self.engine.global_steps
            record["zero_stage"] = self.engine.zero_optimization_stage()
            record["optimizer_cpu_offload"] = bool(
                getattr(self.engine.optimizer, "cpu_offload", False)
            )
            if step == 1 and self.weight_probe is not None:
                record["fp32_master_probe_changed"] = int(
                    (self.weight_probe.cpu() != self.weight_probe_initial).sum()
                )
                record["optimizer_class"] = type(self.engine.optimizer.optimizer).__name__
        if self.cuda:
            gib = 1024**3
            record.update(
                allocated_gib=torch.cuda.memory_allocated() / gib,
                reserved_gib=torch.cuda.memory_reserved() / gib,
                peak_allocated_gib=torch.cuda.max_memory_allocated() / gib,
                peak_reserved_gib=torch.cuda.max_memory_reserved() / gib,
            )
        self.active = False
        self.last_end = end
        if self.trace is not None:
            self.trace.__exit__(None, None, None)
            target = self.output / "profiler"
            target.mkdir(exist_ok=True)
            self.trace.export_chrome_trace(str(target / f"rank0-step{step}.pt.trace.json"))
            self.trace = None
        records = [None] * self.accelerator.num_processes
        if dist.is_initialized():
            dist.all_gather_object(records, record)
        else:
            records = [record]
        if self.writer:
            row = {
                "step": step,
                "ranks": records,
                "warmup": step <= self.options.get("warmup_steps", 5),
                "trace": step == self.trace_step,
            }
            self.records.append(row)
            with (self.output / "timings.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            for name in (
                "update_wall_s",
                "observed_interval_s",
                "peak_allocated_gib",
                "peak_reserved_gib",
                "allocated_gib",
                "reserved_gib",
            ):
                values = [r[name] for r in records if name in r]
                if values:
                    group = "Memory" if name.endswith("gib") else "Timing"
                    self.writer.add_scalar(f"{group}/{name}_rank_max", max(values), step)
                    self.writer.add_scalar(f"{group}/{name}_rank_mean", np.mean(values), step)
            for name, count in records[0]["call_counts"].items():
                self.writer.add_scalar("CallCount/" + name, count, step)
            self.writer.add_scalar(
                "Cache/hit_fraction",
                np.mean([r["input_details"].get("latent_cache_hit", False) for r in records]),
                step,
            )
            for key, group in (
                ("host_call_s", "HostCallSeconds"),
                ("gpu_stream_span_s", "StreamSpanSeconds"),
            ):
                names = sorted(set().union(*(r[key] for r in records)))
                for name in names:
                    self.writer.add_scalar(
                        f"{group}/{name}_rank_max",
                        max(r[key].get(name, 0.0) for r in records),
                        step,
                    )
            self.writer.add_histogram(
                "Timing/update_wall_across_ranks",
                np.array([r["update_wall_s"] for r in records]),
                step,
            )
            self.writer.flush()
            print(
                f"PROFILE_UPDATE step={step} wall_max={max(r['update_wall_s'] for r in records):.3f}s peak_allocated_max={max(r.get('peak_allocated_gib', 0.) for r in records):.2f}GiB",
                flush=True,
            )

    def close(self):
        if not self.enabled or not self.writer:
            return
        steady = [row for row in self.records if not row["warmup"] and not row["trace"]]
        summary = {
            "world_size": self.accelerator.num_processes,
            "warmup_steps": self.options.get("warmup_steps", 5),
            "steady_steps": len(steady),
        }
        for metric in (
            "update_wall_s",
            "observed_interval_s",
            "peak_allocated_gib",
            "peak_reserved_gib",
        ):
            values = [
                max(r[metric] for r in row["ranks"]) for row in steady if metric in row["ranks"][0]
            ]
            if values:
                summary[metric] = dict(
                    zip(
                        ("mean", "p50", "p90", "p95", "min", "max"),
                        [
                            float(np.mean(values)),
                            *map(float, np.percentile(values, [50, 90, 95])),
                            float(min(values)),
                            float(max(values)),
                        ],
                    )
                )
                self.writer.add_histogram(
                    "Distribution/" + metric, np.asarray(values), len(self.records)
                )
        stages = {}
        for row in steady:
            names = set().union(*(r["gpu_stream_span_s"] for r in row["ranks"]))
            for name in names:
                stages.setdefault(name, []).append(
                    max(r["gpu_stream_span_s"].get(name, 0.0) for r in row["ranks"])
                )
        summary["stage_stream_span_mean_s"] = {k: float(np.mean(v)) for k, v in stages.items()}
        summary["stage_stream_span_distribution_s"] = {}
        for name, values in stages.items():
            summary["stage_stream_span_distribution_s"][name] = dict(
                mean=float(np.mean(values)),
                p50=float(np.percentile(values, 50)),
                p95=float(np.percentile(values, 95)),
                min=float(min(values)),
                max=float(max(values)),
            )
            self.writer.add_histogram(
                "StageDistribution/" + name, np.asarray(values), len(self.records)
            )
        summary["data_wait_mean_s"] = (
            float(
                np.mean(
                    [
                        max(r["host_call_s"].get("data_wait", 0.0) for r in row["ranks"])
                        for row in steady
                    ]
                )
            )
            if steady
            else None
        )
        (self.output / "profile_summary.json").write_text(json.dumps(summary, indent=2))
        self.writer.flush()
