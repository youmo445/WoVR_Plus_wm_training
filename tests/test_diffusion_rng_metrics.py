import types
import unittest
import torch
from wovrplus.training.random_streams import DiffusionRandomStreams, stream_seed
from wovrplus.training.loss_metrics import LossMetrics
from wovrplus.training.loss import world_model_loss
from wovrplus.schedulers.flow_match import FlowMatchScheduler


class DiffusionRNGTests(unittest.TestCase):
    def draw(self, stream):
        with stream.noise_context():
            noise = torch.randn(64)
        return noise, stream.randint(0, 1000, (32,))

    def test_rank_independence_and_repeatability(self):
        samples = [self.draw(DiffusionRandomStreams(42, rank, "cpu")) for rank in range(16)]
        self.assertEqual(len({x.numpy().tobytes() for x, _ in samples}), 16)
        self.assertEqual(len({t.numpy().tobytes() for _, t in samples}), 16)
        for rank, expected in enumerate(samples):
            actual = self.draw(DiffusionRandomStreams(42, rank, "cpu"))
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_loader_global_rng_and_validation_do_not_advance_training_stream(self):
        train = DiffusionRandomStreams(42, 3, "cpu")
        control = DiffusionRandomStreams(42, 3, "cpu")
        self.draw(train); self.draw(control)
        torch.manual_seed(125)
        before = torch.get_rng_state().clone()
        self.draw(DiffusionRandomStreams(42, 3, "cpu", "validation", 2))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        # Arbitrary data-loader/global draws do not influence diffusion streams.
        torch.randperm(1000)
        a, b = self.draw(train), self.draw(control)
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        self.assertNotEqual(stream_seed(42, 3, "train", "noise"),
                            stream_seed(42, 3, "validation", "noise"))

    def test_context_restores_global_state_on_exception(self):
        stream = DiffusionRandomStreams(42, 0, "cpu")
        before = torch.get_rng_state().clone()
        with self.assertRaises(RuntimeError):
            with stream.noise_context():
                torch.randn(10)
                raise RuntimeError("intentional test")
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_metrics_use_all_samples_and_count_weighted_rank_merge(self):
        a, b = LossMetrics("cpu"), LossMetrics("cpu")
        # Unequal local counts ensure averaging rank means would fail this test.
        for store, weighted, raw, t in [(a, 2., 4., 0.), (b, 8., 6., .15), (b, 20., 8., 1.)]:
            store.add(torch.tensor(weighted), dict(unweighted_mse=torch.tensor(raw),
                       normalized_timestep=torch.tensor(t)))
        report = LossMetrics.summarize(a.sums + b.sums)
        self.assertEqual(report["sample_count"], 3)
        self.assertEqual(report["weighted_loss"], 10.)
        self.assertEqual(report["unweighted_mse"], 6.)
        self.assertEqual(set(report["bins"]), {"000-100", "100-200", "900-1000"})
        self.assertEqual(sum(v["sample_count"] for v in report["bins"].values()), 3)
        a.reset(); self.assertEqual(a.sums.sum().item(), 0.)

    def test_unweighted_mse_uses_horizon_mask_and_keeps_objective_gradient(self):
        scheduler = FlowMatchScheduler(shift=5, sigma_min=0., extra_one_step=True)
        scheduler.set_timesteps(1000, training=True)
        parameter = torch.tensor(2., requires_grad=True)
        def predict(input_latents, noise, **kw):
            error = torch.ones_like(noise) * parameter
            error = torch.cat((error[:, :, :2] * 100, error[:, :, 2:]), dim=2)
            return noise - input_latents + error
        pipe = types.SimpleNamespace(scheduler=scheduler, torch_dtype=torch.float32,
                                     device="cpu", model_fn=predict)
        rng = DiffusionRandomStreams(42, 0, "cpu")
        latents = torch.randn(1, 48, 4, 2, 2)
        loss = world_model_loss(pipe, input_latents=latents, noise=torch.randn_like(latents),
                               clean_latent_slots=2, ti2v2_horizon_loss_latent_slots=2,
                               diffusion_rng=rng)
        self.assertAlmostEqual(pipe.last_loss_metrics["unweighted_mse"].item(), 4., places=5)
        t = pipe.last_loss_metrics["normalized_timestep"] * 1000
        expected_weight = scheduler.training_weight(t)
        torch.testing.assert_close(loss, 4 * expected_weight)
        loss.backward()
        torch.testing.assert_close(parameter.grad, (4 * expected_weight).reshape(()))


if __name__ == "__main__":
    unittest.main()
