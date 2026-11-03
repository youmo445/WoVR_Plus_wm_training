"""Two CPU ranks exercise real module RNG, accumulation, interval metrics and eval."""
import argparse
import json
import os
import types
from pathlib import Path
import torch
from accelerate import Accelerator, PartialState
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from wovrplus.training.loop import save_run_config, launch_training_task
from wovrplus.training.module import WorldModelTrainingModule, CheckpointWriter
from wovrplus.training.loss import world_model_loss
from wovrplus.schedulers.flow_match import FlowMatchScheduler

torch.set_num_threads(1)
PartialState(cpu=True, backend="gloo")
accelerator = Accelerator(cpu=True, gradient_accumulation_steps=2)


class Dataset(torch.utils.data.Dataset):
    def __len__(self):
        return 96

    def __getitem__(self, i):
        return torch.full((1, 48, 4, 2, 2), (i % 5) / 5.)


class TinyPipeline(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cpu"
        self.torch_dtype = torch.float32
        self.dit = torch.nn.Linear(1, 1, bias=False)
        self.vae = torch.nn.Identity()
        self.vace = self.text_encoder = None
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0., extra_one_step=True)
        self.training_loss = types.MethodType(world_model_loss, self)

    def prepare_sample(self, data):
        return dict(input_latents=data, noise=torch.randn_like(data))

    def model_fn(self, latents, **kw):
        return latents * self.dit.weight.reshape(())


output = Path(os.environ["WOVR_TEST_OUTPUT"])
args = argparse.Namespace(output_path=str(output), save_steps=6, eval_steps=3,
                          log_steps=5, num_epochs=1, max_steps=12,
                          gradient_accumulation_steps=2, seed=42, init_checkpoint_path=None,
                          find_unused_parameters=False, dataset_num_workers=0,
                          learning_rate=.001, weight_decay=0.)
cfg = dict(model=dict(family="wan_action"), data=dict(context_frames=5, horizon_frames=8),
           training=dict(trainable=["dit"], seed=42, gradient_checkpointing=False,
                         gradient_checkpointing_offload=False, low_noise_offset=1))
save_run_config(args)
model = WorldModelTrainingModule(cfg, TinyPipeline())
launch_training_task(Dataset(), Dataset(), model, CheckpointWriter(output / "checkpoints"), args,
                     accelerator=accelerator)
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    config = json.loads((output / "config.json").read_text())
    seeds = config["runtime"]["diffusion_rng"]["per_rank_seeds"]
    assert len(seeds) == 2 and seeds[0][1] != seeds[1][1] and seeds[0][2] != seeds[1][2]
    rows = [json.loads(l) for l in (output / "loss_metrics.jsonl").read_text().splitlines()]
    train, val = ([r for r in rows if r["phase"] == phase] for phase in ("train", "validation"))
    assert sum(r["sample_count"] for r in train) == 48
    assert train[-1]["optimizer_steps"] == 2 and train[-1]["sample_count"] == 8
    assert len(val) == 2 and all(r["sample_count"] == 6 for r in val)
    assert all(sum(v["sample_count"] for v in r["bins"].values()) == r["sample_count"] for r in rows)
    ea = EventAccumulator(str(output / "tensorboard"), size_guidance={"scalars":0}); ea.Reload()
    assert "Loss/train_interval" in ea.Tags()["scalars"]
    assert "MSE/train_interval_unweighted" in ea.Tags()["scalars"]
    assert "Loss/validation_interval" in ea.Tags()["scalars"]
    assert any(t.startswith("Timestep/train/") for t in ea.Tags()["scalars"])
    assert (output / "checkpoints/step-00000012.safetensors").exists()
    print("DISTRIBUTED_RNG_METRICS_PASS", flush=True)
