from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
from wovrplus.config import ROOT, resolve_config
from wovrplus.data.latent_cache import (
    LatentCacheDataset,
    code_fingerprint,
    digest_file,
    encoding_spec,
    file_identity,
    sample_rng,
)
from wovrplus.pipelines.world_model import WorldModelPipeline


class LatentCacheTests(unittest.TestCase):
    def test_yaml_source_list(self):
        from wovrplus.data.robotwin import _parse_base_paths

        paths = ["/external/adjust_bottle", "/external/adjust_bottle_1"]
        self.assertEqual(_parse_base_paths(paths), [Path(p) for p in paths])
        self.assertEqual(_parse_base_paths(json.dumps(paths)), [Path(p) for p in paths])
        with self.assertRaises(ValueError):
            _parse_base_paths([])

    def test_sampling_rng_isolation(self):
        random.seed(71)
        np.random.seed(71)
        torch.random.default_generator.manual_seed(71)
        state = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
        with patch("torch.cuda.manual_seed_all") as cuda_seed:
            with sample_rng(12):
                a = (random.random(), np.random.rand(), torch.rand(1))
            with sample_rng(12):
                b = (random.random(), np.random.rand(), torch.rand(1))
            cuda_seed.assert_not_called()
        self.assertEqual(a[:2], b[:2])
        self.assertTrue(torch.equal(a[2], b[2]))
        self.assertEqual(random.getstate(), state[0])
        self.assertTrue(np.array_equal(np.random.get_state()[1], state[1][1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), state[2]))

    def test_cached_pipeline_skips_vae_and_resamples_noise(self):
        class VAE(torch.nn.Module):
            calls = 0

            def encode(self, x, **kwargs):
                self.calls += 1
                return torch.nn.functional.interpolate(x, size=(4, 2, 2), mode="nearest").repeat(
                    1, 16, 1, 1, 1
                )

        for family in ("action", "vace"):
            vae = VAE()
            vace = torch.nn.Module() if family == "vace" else None
            if vace is not None:
                vace.vace_in_dim = 48
            pipe = WorldModelPipeline(None, vae, vace=vace, dtype=torch.float32)
            video = [Image.new("RGB", (32, 32), (120, 50, 40)) for _ in range(13)]
            sample = {"video": video, "vace_video": video, "action": torch.ones(13, 14)}
            torch.manual_seed(42)
            online = pipe.prepare_sample(sample)
            keys = ("input_latents", "action" if vace is None else "vace_context")
            cached = {"latent_cache": {k: online[k].clone() for k in keys}}
            calls = vae.calls
            torch.manual_seed(42)
            replay = pipe.prepare_sample(cached)
            for k in keys + ("noise",):
                self.assertTrue(torch.equal(online[k], replay[k]), k)
            self.assertEqual(vae.calls, calls)
            self.assertFalse(torch.equal(replay["noise"], pipe.prepare_sample(cached)["noise"]))
            with self.assertRaisesRegex(ValueError, "future"):
                pipe.generate(cached)

    def test_manifest_integrity_and_config_invalidation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train/tensors").mkdir(parents=True)
            weights = root / "weights"
            weights.mkdir()
            (weights / "Wan2.2_VAE.pth").write_bytes(b"vae")
            source = root / "source.hdf5"
            source.write_bytes(b"original")
            cfg = resolve_config([ROOT / "configs/train/robotwin_vace.yaml"])
            cfg["model"]["model_dir"] = str(weights)
            cfg["data"]["cache"].update(root=str(root), mode="read", verify_checksum=True)
            tensors = {
                "input_latents": torch.ones(1, 48, 4, 2, 2),
                "vace_context": torch.ones(1, 72, 4, 2, 2),
            }
            path = root / "train/tensors/000.safetensors"
            save_file(tensors, str(path), metadata={"signature": "test"})
            record = {
                "file": "tensors/000.safetensors",
                "source_index": 0,
                "variant": 0,
                "seed": 1,
                "sha256": digest_file(path),
                "prompt": "fixed",
                "meta": {},
                "tensors": {
                    k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in tensors.items()
                },
            }
            manifest = {
                "schema": 1,
                "complete": True,
                "encoding_spec": encoding_spec(cfg),
                "code_sha256": code_fingerprint(),
                "vae_identity": file_identity(weights / "Wan2.2_VAE.pth"),
                "sources": [file_identity(source)],
                "signature": "test",
                "records": [record],
            }
            file = root / "train/manifest.json"
            file.write_text(json.dumps(manifest))
            ds = LatentCacheDataset(cfg, "train")
            self.assertEqual(len(ds), 1)
            self.assertEqual(set(ds[0]["latent_cache"]), set(tensors))
            changed = deepcopy(cfg)
            changed["data"]["sampling"]["context_sphere_blur_prob"] = 0
            with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                LatentCacheDataset(changed, "train")
            source.write_bytes(b"changed data")
            with self.assertRaisesRegex(ValueError, "source changed"):
                LatentCacheDataset(cfg, "train")
            path.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "Corrupt"):
                ds[0]
            manifest["complete"] = False
            file.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                LatentCacheDataset(cfg, "train")


if __name__ == "__main__":
    unittest.main()
