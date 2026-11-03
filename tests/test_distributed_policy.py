import json
import tempfile
import unittest
from pathlib import Path
import yaml
from wovrplus.training.distributed import reject_optimizer_offload, check_launch_config


class DistributedPolicyTest(unittest.TestCase):
    def test_optimizer_offload_rejected(self):
        for device in ("cpu", "CPU", "nvme"):
            with self.subTest(device=device), self.assertRaisesRegex(
                ValueError, "offload is disabled"
            ):
                reject_optimizer_offload(
                    {"zero_optimization": {"offload_optimizer": {"device": device}}}
                )
        for config in (
            {},
            {"zero_optimization": {"stage": 0}},
            {"zero_optimization": {"offload_optimizer": {"device": "none"}}},
        ):
            reject_optimizer_offload(config)

    def test_external_launch_file_cannot_enable_offload(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds, launch = Path(tmp) / "ds.json", Path(tmp) / "launch.yaml"
            ds.write_text(
                json.dumps({"zero_optimization": {"offload_optimizer": {"device": "cpu"}}})
            )
            launch.write_text(
                yaml.safe_dump({"deepspeed_config": {"deepspeed_config_file": str(ds)}})
            )
            with self.assertRaises(ValueError):
                check_launch_config(launch)
            launch.write_text(
                yaml.safe_dump({"deepspeed_config": {"offload_optimizer_device": "cpu"}})
            )
            with self.assertRaises(ValueError):
                check_launch_config(launch)


if __name__ == "__main__":
    unittest.main()
