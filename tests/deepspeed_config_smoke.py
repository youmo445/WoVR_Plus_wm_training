"""Tiny actual-device test of the supplied Accelerate/DeepSpeed launch configuration."""

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

accelerator = Accelerator()
model = torch.nn.Linear(128, 128)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loader = DataLoader([torch.ones(128) for _ in range(32)], batch_size=1)
model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
zero = accelerator.deepspeed_config["zero_optimization"]
assert model.zero_optimization_stage() == zero["stage"]
assert getattr(model.optimizer, "partition_gradients", False) == (zero["stage"] == 2)
offload = zero["offload_optimizer"]["device"]
assert not getattr(model.optimizer, "cpu_offload", False)
assert offload == "none"
for index, batch in enumerate(loader):
    with accelerator.accumulate(model):
        loss = model(batch.to(torch.bfloat16)).square().mean()
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
    if index == 2:
        break
assert model.global_steps == 3
accelerator.wait_for_everyone()
print(
    f"DEEPSPEED_CONFIG_PASS rank={accelerator.process_index} steps={model.global_steps} "
    f"zero_stage={model.zero_optimization_stage()} "
    f"partition_gradients={getattr(model.optimizer, 'partition_gradients', False)} "
    f"cpu_offload={getattr(model.optimizer, 'cpu_offload', False)}",
    flush=True,
)
