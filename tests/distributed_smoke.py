"""Run with torchrun --standalone --nproc_per_node=2 tests/distributed_smoke.py.

CPU/gloo test of the real training/checkpoint/evaluation loop. No production job.
"""

import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch
import torch
from accelerate import Accelerator, PartialState
from wovrplus.training.loop import save_run_config, launch_training_task
from wovrplus.training.module import CheckpointWriter

torch.set_num_threads(1)
PartialState(cpu=True, backend="gloo")


class Dataset(torch.utils.data.Dataset):
    def __len__(self):
        return 40

    def __getitem__(self, i):
        return torch.tensor([float(i % 3)])


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pipe = torch.nn.Module()
        self.pipe.dit = torch.nn.Linear(1, 1)

    def trainable_modules(self):
        return self.parameters()

    def forward(self, data):
        return (self.pipe.dit(data) - data).square().mean()

    def export_trainable_state_dict(self, state):
        return state


output = Path(os.environ["WOVR_TEST_OUTPUT"])
args = argparse.Namespace(
    output_path=str(output),
    save_steps=2,
    eval_steps=3,
    log_steps=1,
    num_epochs=1,
    max_steps=3,
    gradient_accumulation_steps=2,
    seed=42,
    init_checkpoint_path=None,
    find_unused_parameters=False,
    dataset_num_workers=0,
    learning_rate=0.01,
    weight_decay=0.0,
)
save_run_config(args)
with patch(
    "wovrplus.training.loop.Accelerator", side_effect=lambda **kw: Accelerator(cpu=True, **kw)
):
    launch_training_task(
        Dataset(), Dataset(), Model(), CheckpointWriter(str(output / "checkpoints")), args
    )
if int(os.environ["RANK"]) == 0:
    config = json.loads((output / "config.json").read_text())
    assert config["runtime"]["world_size"] == 2
    assert config["runtime"]["global_batch_size"] == 4
    assert (output / "checkpoints/step-00000002.safetensors").is_file()
    assert (output / "checkpoints/step-00000003-final.safetensors").is_file()
    print("DISTRIBUTED_TRAIN_CHECKPOINT_EVAL_PASS", flush=True)
