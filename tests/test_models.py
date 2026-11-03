import copy
import os
import time
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from einops import rearrange
from safetensors.torch import save_file
from wovrplus.models.wan_dit import WanModel, sinusoidal_embedding_1d
from wovrplus.models.action import WanActionModel
from wovrplus.models.vace import VaceWanModel
from wovrplus.models.loading import load_world_checkpoint
from wovrplus.pipelines.model_fn import model_fn_wan
from wovrplus.pipelines.world_model import WorldModelPipeline
from wovrplus.pipelines.vace_conditions import VACEConditionEncoder
from wovrplus.training.module import WorldModelTrainingModule
from wovrplus.config import resolve_config, ROOT

torch.set_num_threads(1)
SMALL = dict(
    dim=48,
    in_dim=48,
    ffn_dim=64,
    out_dim=48,
    text_dim=32,
    freq_dim=16,
    eps=1e-6,
    patch_size=(1, 2, 2),
    num_heads=4,
    num_layers=2,
    has_image_input=False,
    seperated_timestep=True,
    require_vae_embedding=False,
    require_clip_embedding=False,
)


def reference(name):
    code = (Path(__file__).parent / "fixtures" / f"{name}_forward.txt").read_text()
    namespace = dict(
        torch=torch,
        np=np,
        os=os,
        time=time,
        rearrange=rearrange,
        sinusoidal_embedding_1d=sinusoidal_embedding_1d,
    )
    namespace["LongCatVideoTransformer3DModel"] = type("UnusedLongCat", (), {})
    exec("from __future__ import annotations\n" + code, namespace)
    return namespace["model_fn_wan_video"]


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)

    def test_three_action_modes_match_rlinf_outputs_and_gradients(self):
        original = reference("rlinf")
        for mode in ("cross_attn", "adaln", "cross_attn_and_adaln"):
            for batch in (1, 2):
                with self.subTest(mode=mode, batch=batch):
                    model = WanActionModel(**SMALL, action_dim=7, action_mode=mode)
                    model.has_action_mode = True
                    model.one_frame_condition = False
                    model.five_frame_condition = True
                    x = torch.randn(batch, 48, 4, 4, 4)
                    actions = torch.randn(batch, 13, 7)
                    given = actions[0] if batch == 1 else actions
                    actual = model_fn_wan(model, x, torch.tensor([500.0]), action=given)
                    # Upstream batched cross-attn path references undefined B. Compare it per item.
                    if batch == 2 and mode == "cross_attn":
                        expected = torch.cat(
                            [
                                original(
                                    dit=model,
                                    latents=x[i : i + 1],
                                    timestep=torch.tensor([500.0]),
                                    action=actions[i],
                                    fuse_vae_embedding_in_latents=True,
                                )
                                for i in range(batch)
                            ]
                        )
                    else:
                        expected = original(
                            dit=model,
                            latents=x,
                            timestep=torch.tensor([500.0]),
                            action=given,
                            fuse_vae_embedding_in_latents=True,
                        )
                    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
                    params = [p for p in model.parameters() if p.requires_grad]
                    g1 = torch.autograd.grad(
                        actual.square().mean(), params, retain_graph=True, allow_unused=True
                    )
                    g2 = torch.autograd.grad(expected.square().mean(), params, allow_unused=True)
                    for a, b in zip(g1, g2):
                        self.assertEqual(a is None, b is None)
                        if a is not None:
                            torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-6)
                    changed = model_fn_wan(model, x, torch.tensor([500.0]), action=given + 1)
                    self.assertGreater((actual - changed).abs().max().item(), 1e-5)
                    with self.assertRaises(ValueError):
                        model_fn_wan(model, x, torch.tensor([500.0]), action=torch.zeros(13, 8))

    def test_action_checkpoint_keys_and_round_trip(self):
        for mode in ("crossattn", "modulation", "both"):
            pipe = torch.nn.Module()
            pipe.dit = WanActionModel(**SMALL, action_dim=7, action_mode=mode)
            keys = pipe.dit.state_dict()
            self.assertEqual(any(k.startswith("action_mlp1.") for k in keys), mode != "modulation")
            self.assertEqual(any(k.startswith("action_mlp2.") for k in keys), mode != "crossattn")
            with tempfile.TemporaryDirectory() as temp:
                path = str(Path(temp) / "model.safetensors")
                save_file(keys, path)
                expected = copy.deepcopy(keys)
                with torch.no_grad():
                    for p in pipe.parameters():
                        p.zero_()
                load_world_checkpoint(pipe, path)
                for key, value in expected.items():
                    torch.testing.assert_close(value, pipe.dit.state_dict()[key])
                save_file({"wrong": torch.ones(1)}, path)
                with self.assertRaises(ValueError):
                    load_world_checkpoint(pipe, path)

    def test_vace_four_channel_layouts_match_previous_pipeline(self):
        original = reference("vace")
        for channels in (48, 54, 72, 144):
            with self.subTest(channels=channels):
                dit = WanModel(**SMALL)
                vace = VaceWanModel(
                    vace_layers=(0,),
                    vace_in_dim=channels,
                    dim=48,
                    num_heads=4,
                    ffn_dim=64,
                    view_embedding_num_views=3,
                )
                vace.init_from_dit(dit)
                x = torch.randn(1, 48, 4, 6, 4)
                ctx = torch.randn(1, 5, 32)
                control = torch.randn(1, channels, 4, 6, 4)
                kwargs = dict(
                    dit=dit,
                    latents=x,
                    timestep=torch.tensor([500.0]),
                    context=ctx,
                    vace=vace,
                    vace_context=control,
                    clean_latent_slots=2,
                )
                actual = model_fn_wan(**kwargs)
                expected = original(**kwargs, fuse_vae_embedding_in_latents=True)
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
                actual.square().mean().backward()
                self.assertIsNotNone(vace.vace_patch_embedding.weight.grad)
                self.assertTrue(torch.isfinite(vace.vace_patch_embedding.weight.grad).all())

    def test_condition_encoder_layout(self):
        encoder = VACEConditionEncoder()
        act = torch.randn(1, 48, 4, 6, 4)
        for channels in (48, 54, 72, 144):
            pipe = SimpleNamespace(
                vae=SimpleNamespace(z_dim=48),
                vace=SimpleNamespace(vace_in_dim=channels),
                device="cpu",
                torch_dtype=torch.float32,
                load_models_to_device=lambda names: None,
            )
            c = {48: 0, 54: 3, 72: 12, 144: 48}[channels]
            rays = torch.randn(1, c, 4, 6, 4) if c else None
            output = encoder.process(
                pipe, act, None, rays, rays, 1.0, 96, 64, 13, False, (34, 34), (18, 16)
            )
            self.assertEqual(tuple(output["vace_context"].shape), (1, channels, 4, 6, 4))

    def test_pipeline_loss_and_generation_with_context1_and_context5(self):
        from PIL import Image

        class TinyVAE(torch.nn.Module):
            def encode(self, x, **kw):
                return torch.nn.functional.interpolate(
                    x, size=((x.shape[2] + 3) // 4, 2, 2), mode="nearest"
                ).repeat(1, 16, 1, 1, 1)

            def decode(self, x, **kw):
                return torch.nn.functional.interpolate(
                    x[:, :3], size=((x.shape[2] - 1) * 4 + 1, 32, 32), mode="nearest"
                )

        for context in (1, 5):
            cfg = resolve_config(
                [ROOT / "configs/train/robotwin_action.yaml"],
                ["model.dtype=float32", f"data.context_frames={context}"],
            )
            pipe = WorldModelPipeline(
                WanActionModel(**SMALL, action_dim=14), TinyVAE(), dtype=torch.float32
            )
            module = WorldModelTrainingModule(cfg, pipe)
            count = context + 8
            sample = {
                "video": [Image.new("RGB", (32, 32), (100, 50, 20)) for _ in range(count)],
                "action": torch.randn(count, 14),
            }
            loss = module(sample)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            video = pipe.generate(sample, context_frames=context, steps=2)
            self.assertEqual(len(video), count)
            self.assertTrue(pipe.scheduler.training)


if __name__ == "__main__":
    unittest.main()
