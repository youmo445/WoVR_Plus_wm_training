# Release validation

Validation date: **2026-11-03**. These checks describe the public source tree, not a claim that every accelerator or augmentation combination has been benchmarked.

## Checks performed

- **24 CPU unit tests passed**: model forward/gradient parity against retained references, all three vector injection modes, four VACE channel layouts, additive mesh patch injection, cache/replay fixtures, per-rank diffusion RNG isolation, loss interval/bin accounting, optimizer-offload rejection, a complete 503-update training-loop fixture with checkpoint/evaluation, external path resolution, online mesh sample adaptation, paired online/cached recipes, and legacy import/family aliases.
- All six canonical training scripts passed configuration dry-runs. Raymap + VACE, MotionImage + Patch Addition and vector + AdaLN each provide an online and a cached entry. The default vector injection is AdaLN; MotionImage retains its additive patch equation and checkpoint tensor names.
- **All three families passed a real-asset preflight** using the five published HDF5 groups and the public RoboTwin embodiment download. The check matched all **825 original WAN parameter keys and shapes**, discovered **84,864 training / 21,899 validation windows**, and read training/validation samples with **13 frames at 320×768** stacked geometry. Vector actions have shape **13×14**. Mesh motion images match the RGB frame count and canvas dimensions.
- The robot asset download was exercised against `TianxingChen/RoboTwin2.0` revision `3dc3b798668feb99ac61cc9086d84cbcc3d79186`. The downloaded configuration and URDF matched the geometry used to prepare the training pipeline.
- All **500 published HDF5 files** passed streaming SHA256 verification against the manifest. Release preparation also compared every dataset array with its source; total payload is **5,676,701,632 bytes**, **121,263 timesteps**. Internal reference-path attributes were normalized, not data arrays.
- Shell syntax checks and Python package wheel build passed. Credentials, machine-specific training paths, model weights and dataset payloads are excluded from the source repository.

## Limits

The preflight inspects model headers and real data without allocating the complete trainable 5B model. It does **not** establish a fresh full-scale throughput, memory, or convergence result for this cleaned checkout. CPU tests use small model fixtures. No new full-window cache or long distributed training run was launched solely for publication, and active experiments were not modified.

Caches must be regenerated with this release's preprocessing fingerprint and final asset paths. Test the intended accelerator setup with a short run before committing a large allocation. Record the code commit, resolved config, asset revisions and profiling settings alongside results.

## 中文

本次发布通过了 24 项 CPU 测试、六个在线/缓存入口的 dry-run、三类方法的真实权重头部/数据预检查、500 个 HDF5 的 SHA256 校验、shell 语法检查与 wheel 构建。命名更新额外核验了在线/缓存配置配对以及旧导入/方法名兼容入口；vector 默认 AdaLN，MotionImage 保持 Patch Addition 和原参数键。预检查匹配全部 825 个原始 WAN 参数键/形状，识别训练 84,864、验证 21,899 个窗口，读取的样本均为 13 帧、三视角拼接 320×768；向量为 13×14，网格画布与 RGB 对齐。

这些结果不等同于针对发布版重新完成了 16 卡长训练或吞吐测试。本次没有为发布重复构建全量缓存，也没有修改正在运行的训练。使用公开版本时需按新的代码指纹与最终资产路径生成缓存，并对目标硬件先进行短程验证。
