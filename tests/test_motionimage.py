import unittest
import torch
from torch.nn import functional as F
from wovrplus.models.motionimage import WanMotionImageModel
from wovrplus.pipelines.model_fn import model_fn_wan
from wovrplus.data.condition_cache import motion_key
from test_models import SMALL


class MotionImageTests(unittest.TestCase):
    def test_additive_replay_patch_equation_and_gradient(self):
        torch.manual_seed(7)
        model = WanMotionImageModel(**SMALL)
        x = torch.randn(1, 48, 4, 4, 4)
        motion = torch.randn_like(x)
        context = torch.randn(1, 4, 32)
        prediction = model_fn_wan(model, x, torch.tensor([500.]), context=context, replay_latents=motion)
        # Verify the reference equation through a hook on the first transformer.
        observed = []
        hook = model.blocks[0].register_forward_pre_hook(lambda module, args: observed.append(args[0].detach()))
        model_fn_wan(model, x, torch.tensor([500.]), context=context, replay_latents=motion)
        hook.remove()
        expected = (model.patch_embedding(x) + model.replay_embedding(motion)).flatten(2).transpose(1, 2)
        torch.testing.assert_close(observed[0], expected)
        prediction.square().mean().backward()
        self.assertTrue(torch.isfinite(model.replay_embedding.weight.grad).all())
        self.assertGreater(model.replay_embedding.weight.grad.abs().max().item(), 0)
        changed = model_fn_wan(model, x, torch.tensor([500.]), context=context, replay_latents=motion + 1)
        self.assertGreater((prediction - changed).abs().max().item(), 1e-5)
        with self.assertRaises(ValueError):
            model_fn_wan(model, x, torch.tensor([500.]), context=context)

    def test_condition_identity_depends_on_order_and_duplicate_frames(self):
        meta = dict(path="episode0.hdf5", frame_ids=[0, 0, 4, 7], camera_names=["head", "left", "right"])
        # RGB blur must not cause duplicate motion encodings.
        self.assertEqual(motion_key(meta), motion_key({**meta, "context_sphere_blur": {"sigma": 7}}))
        self.assertNotEqual(motion_key(meta), motion_key({**meta, "frame_ids": [0, 4, 7]}))
        self.assertNotEqual(motion_key(meta), motion_key({**meta, "camera_names": ["left", "head", "right"]}))
