import torch


def world_model_loss(self, **inputs):
    max_timestep_boundary = int(
        inputs.get("max_timestep_boundary", 1) * self.scheduler.num_train_timesteps
    )
    min_timestep_boundary = int(
        inputs.get("min_timestep_boundary", 0) * self.scheduler.num_train_timesteps
    )
    rng = inputs.get("diffusion_rng")
    randint = torch.randint if rng is None else rng.randint
    timestep_id = randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.torch_dtype, device=self.device)

    input_latents = inputs["input_latents"]
    noise = inputs["noise"]
    clean_slots = int(inputs.get("clean_latent_slots", 2))
    if clean_slots not in (1, 2) or input_latents.shape[2] <= clean_slots:
        raise ValueError(
            f"Wan2.2 TI2V2 VACE expects at least 2 latent slots, got {tuple(input_latents.shape)}."
        )

    latents = self.scheduler.add_noise(input_latents, noise, timestep)
    latents[:, :, 0:1] = input_latents[:, :, 0:1]

    max_low_noise_step_offset = int(inputs.get("ti2v2_low_noise_step_offset", 50))
    actual_low_noise_step_offset = int(randint(0, max_low_noise_step_offset + 1, (1,)).item())
    if clean_slots == 1:
        pass  # The second latent slot is a prediction target, not history.
    elif actual_low_noise_step_offset == 0:
        latents[:, :, 1:2] = input_latents[:, :, 1:2]
    else:
        low_noise_idx = torch.tensor(
            [max(0, len(self.scheduler.timesteps) - actual_low_noise_step_offset)]
        )
        low_noise_timestep = self.scheduler.timesteps[low_noise_idx].to(
            dtype=self.torch_dtype, device=self.device
        )
        latents[:, :, 1:2] = self.scheduler.add_noise(
            input_latents[:, :, 1:2],
            noise[:, :, 1:2],
            low_noise_timestep,
        )
    inputs["latents"] = latents

    training_target = self.scheduler.training_target(input_latents, noise, timestep)
    noise_pred = self.model_fn(**inputs, timestep=timestep)

    loss = torch.nn.functional.mse_loss(
        noise_pred.float(), training_target.float(), reduction="none"
    )
    horizon_slots = int(inputs.get("ti2v2_horizon_loss_latent_slots", 4))
    if horizon_slots <= 0 or horizon_slots != loss.shape[2] - clean_slots:
        raise ValueError(
            f"Invalid horizon latent slots {horizon_slots} for loss shape {tuple(loss.shape)}."
        )
    mask = torch.zeros_like(loss)
    mask[:, :, -horizon_slots:] = 1.0
    unweighted_mse = (loss * mask).sum() / mask.sum().clamp_min(1)
    loss = unweighted_mse * self.scheduler.training_weight(timestep)
    self.last_loss_metrics = {
        "unweighted_mse": unweighted_mse.detach(),
        "normalized_timestep": timestep.detach().float() / self.scheduler.num_train_timesteps,
        "timestep_id": int(timestep_id.item()),
    }
    return loss
