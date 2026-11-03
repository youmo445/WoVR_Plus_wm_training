import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from accelerate import Accelerator
from wovrplus.training.loop import launch_training_task, save_run_config


class TrainingTests(unittest.TestCase):
    def test_full_epoch_accumulation_checkpoint_eval(self):
        # 1006 microbatches -> 503 optimizer updates: catches the old 501-batch exit.
        class Data(torch.utils.data.Dataset):
            def __len__(self):
                return 1006

            def __getitem__(self, i):
                return torch.tensor(float(i % 3))

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
                self.seen = 0
                self.eval_seen = []

            def trainable_modules(self):
                return self.parameters()

            def forward(self, data):
                if self.training:
                    self.seen += 1
                else:
                    self.eval_seen.append(float(data))
                return (self.weight - data).square()

        class Writer:
            def __init__(self):
                self.saved = []

            def save_model(self, accelerator, model, name):
                self.saved.append((name, accelerator.unwrap_model(model).seen))

        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                save_steps=250,
                eval_steps=7,
                log_steps=1000,
                num_epochs=1,
                max_steps=0,
                gradient_accumulation_steps=2,
                seed=42,
                output_path=tmp,
                init_checkpoint_path=None,
                find_unused_parameters=False,
                dataset_num_workers=0,
                learning_rate=0.01,
                weight_decay=0.0,
            )
            save_run_config(args)
            model = Model()
            writer = Writer()
            with patch(
                "wovrplus.training.loop.Accelerator",
                side_effect=lambda **kw: Accelerator(cpu=True, **kw),
            ):
                launch_training_task(Data(), Data(), model, writer, args)
            self.assertEqual(model.seen, 1006)
            self.assertEqual(len(model.eval_seen), 14)
            self.assertEqual(
                writer.saved,
                [
                    ("step-00000250.safetensors", 500),
                    ("step-00000500.safetensors", 1000),
                    ("step-00000503-final.safetensors", 1006),
                ],
            )
            self.assertTrue((Path(tmp) / "source_snapshot.tar.gz").exists())
            config = json.loads((Path(tmp) / "config.json").read_text())
            self.assertEqual(config["runtime"]["batches_per_full_epoch"], 1006)
            self.assertEqual(config["runtime"]["global_batch_size"], 2)


if __name__ == "__main__":
    unittest.main()
