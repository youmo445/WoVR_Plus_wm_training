# WoVR+ 世界模型训练

**面向机器人动作条件世界模型的 WAN 2.2 训练代码。**

[English](README.md) · [公开数据集](https://huggingface.co/datasets/jzndd/RoboTwin_adjust_bottle_For_ACWM_Training) · [WoVR+ 项目](https://youmo445.github.io/WoVR_Plus.github.io/) · [Apache-2.0](LICENSE)

本仓库在相同 RoboTwin 观测上提供三类动作表征与注入方式：动作投影图 + VACE、关节向量 + 注意力/调制、PhysCanvas 风格的网格运动图。包含训练、验证集片段生成、可复现 VAE 缓存、分布式优化和 loss 诊断，不依赖同级 DiffSynth-Studio 目录。

## 目录

- [支持的方法](#支持的方法)
- [安装](#安装)
- [下载数据与资产](#下载数据与资产)
- [快速检查](#快速检查)
- [VAE 缓存](#vae-缓存)
- [三组训练入口](#三组训练入口)
- [参数与变体](#参数与变体)
- [输出、权重与验证](#输出权重与验证)
- [Loss 指标与性能分析](#loss-指标与性能分析)
- [推理](#推理)
- [代码结构与测试](#代码结构与测试)
- [范围、许可与致谢](#范围许可与致谢)

## 支持的方法

| 方法 | 动作表征 | 注入方式 | 运行脚本 |
|---|---|---|---|
| **WAN + VACE** | 经正向运动学投影的彩色动作图，可附加标定相机射线 | VACE 条件分支，Transformer 块由 WAN DiT 初始化 | `scripts/train_vace.sh` |
| **WAN + 关节向量** | 14 维：左臂 6 + 左夹爪 + 右臂 6 + 右夹爪 | `cross_attn`、`adaln` 或 `cross_attn_and_adaln` | `scripts/train_vector.sh` |
| **WAN + PhysCanvas 风格运动图** | 把机器人 link 网格经 FK 投影到灰度画布，再由 WAN VAE 编码 | 独立 Conv3d motion patch embedding 与 RGB latent patch embedding **相加** | `scripts/train_physcanvas.sh` |

三类方法均从 **`Wan-AI/Wan2.2-TI2V-5B` 原始权重**开始，新动作投影层单独初始化。PhysCanvas 风格分支仅将其动作表征与注入思路适配到 RoboTwin，不等同于完整复现该项目的视觉、触觉、状态世界模型或策略评估系统。

默认采用三视角、5 帧历史、8 帧未来，每个相机缩放为 256×320。**先在 RGB 空间纵向拼接三路相机，再经过 VAE**，拼接后为 768×320。该流程不是“每视角独立 VAE 编码后拼 latent”；两者在视角边界附近并不等价。模型比较、缓存生成与推理应保持相同布局。

## 安装

使用 Linux、Python ≥3.10，并先安装匹配驱动/加速器的 PyTorch ≥2.4 与 torchvision。完整 WAN 5B 微调需要支持 BF16 的加速器及足够主机内存；分布式训练使用 DeepSpeed。可选安装兼容的 FlashAttention，否则使用 PyTorch SDPA。使用厂商适配环境时，不要用不匹配的 CUDA wheel 覆盖已有 PyTorch。

```bash
git clone https://github.com/youmo445/WoVR_Plus_wm_training.git
cd WoVR_Plus_wm_training
python -m venv .venv
source .venv/bin/activate
# 先按机器环境安装 torch / torchvision。
pip install -e '.[distributed,test]'
```

若存在 `/usr/local/PPU_SDK/envsetup.sh`，运行脚本会加载 PPU SDK。所有脚本默认使用当前激活环境中的 `python`，也可设置 `PYTHON=/path/to/python`。卡数由 `NUM_PROCESSES` 控制。`--dry-run` 仅检查配置，不代表显存足够；完整 5B 模型并非任意单卡都能训练。

## 下载数据与资产

将数据、权重、机器人资产、缓存和输出统一放在**代码仓库之外**，并在缓存生成、训练、推理时使用同一绝对路径。

```bash
export WOVR_ASSET_ROOT=/your/storage/wovr-assets
python scripts/download_assets.py data
python scripts/download_assets.py wan
python scripts/download_assets.py robot
python scripts/verify_dataset.py "$WOVR_ASSET_ROOT/datasets/adjust_bottle"
```

下载脚本会将 Hub revision 解析为 commit，并保存每项资产的 provenance JSON。可分别通过 `--revision COMMIT` 固定版本。仓库不保存访问凭证。

```text
$WOVR_ASSET_ROOT/
├── datasets/adjust_bottle/
│   ├── adjust_bottle/episode0.hdf5 ... episode99.hdf5
│   ├── adjust_bottle_1/ ... adjust_bottle_4/
│   ├── manifest.json
│   └── splits/{train,validation}.txt
├── models/Wan2.2-TI2V-5B/
│   ├── diffusion_pytorch_model-*-of-*.safetensors
│   ├── Wan2.2_VAE.pth
│   ├── models_t5_umt5-xxl-enc-bf16.pth
│   └── google/umt5-xxl/
├── robot/aloha-agilex/{config.yml,urdf/,meshes/,...}
├── cache/
└── outputs/
```

公开数据包含 **500 条 episode、121,263 个时间步、5.68 GB HDF5**。无后缀的 `adjust_bottle` 是 ODE-rollout，`_1`–`_4` 是 SDE-rollout。作者给定配置为 chunkwise-noise、level=0.2、steps=5；文件已核验字段与作者补充说明的区别见[数据卡](https://huggingface.co/datasets/jzndd/RoboTwin_adjust_bottle_For_ACWM_Training)。图像是仿真观测，成功和失败轨迹都保留。

每个目录的 episode 0–79 用于训练，80–99 用于验证，共 **400/100 条轨迹、80/20 个场景**。相同编号在不同 rollout 组中共享场景种子，不应随机按文件拆分，以免场景泄漏。

机器人资产从 RoboTwin 官方 `embodiments.zip` 单独下载，仅提取 `aloha-agilex`。模型权重、机器人资产与数据分别遵循其原有许可。手动目录设置与版本记录见 [docs/assets.md](docs/assets.md)。

## 快速检查

不读取大权重或数据，先检查三组配置：

```bash
bash scripts/train_vace.sh --dry-run
bash scripts/train_vector.sh --dry-run --set model.action_mode=adaln
bash scripts/train_physcanvas.sh --dry-run
```

缓存生成前，可读取原始数据进行预检查：

```bash
bash scripts/train_vace.sh --validate-only --set data.cache.mode=off
```

预检查核对原始 WAN 的全部 825 个参数键/形状，并读取一个训练和一个验证样本，不分配完整可训练模型。因此它不验证显存是否足够。向量与网格默认脚本需要对应条件缓存，应在以下步骤完成后再执行 `--validate-only`。

## VAE 缓存

三种对照方法共享 RGB 窗口和增强计划。先构建一份 **VACE 格式 RGB/动作缓存**，再构建向量与网格条件缓存：

```bash
# 遍历全部窗口，每窗口四个固定增强版本；不同 rank 分片并行处理。
NUM_PROCESSES=8 bash scripts/cache_latents.sh \
  --config configs/train/adjust_bottle_cached_100k.yaml --variants 4

# 读取共享计划中精确 frame_ids 对应的关节向量。
NUM_PROCESSES=8 bash scripts/cache_conditions.sh \
  --config configs/train/adjust_bottle_vector_cached_100k.yaml

# FK 网格灰度画布 -> VAE motion latent。
NUM_PROCESSES=8 bash scripts/cache_conditions.sh \
  --config configs/train/adjust_bottle_physcanvas_cached_100k.yaml
```

全窗口缓存可能需要数小时及显著大于原始压缩数据的存储空间。每个 rank 处理互不重叠的分片，只有完成覆盖核验才写入 `complete` manifest。中断后可用原配置恢复并复用已核验文件，不要同时启动两个写同一目录的 builder。

四个版本固定的是窗口采样和数据增强，**不缓存扩散噪声或 timestep**；这两者仍按 rank、训练更新独立随机采样。缓存训练省去 RGB/动作 VAE 编码；网格路径需要额外预计算 motion latent。VACE 和网格模型仍使用文本编码器；向量配置默认关闭文本。

读取器检查预处理代码指纹、源文件身份、条件签名、张量形状和 dtype。`data.cache.verify_checksum=true` 可额外逐次核验 SHA256。修改路径、预处理代码、视角布局、分辨率、采样或增强后，需要对应的新缓存，通常应指定新的缓存目录重新构建。不要直接复用其他代码版本的内部缓存。

在线编码可使用 `configs/train/robotwin_vace.yaml`、`robotwin_action.yaml` 或 `robotwin_physcanvas.yaml`，将 `data.base_path` 指向五个目录并保持 `data.cache.mode=off`。做在线/缓存数值对照时，必须使用相同采样计划。`scripts/validate_latent_cache.py` 专门用于 VACE 在线重放与缓存的一致性检查。

## 三组训练入口

相关缓存完整后运行：

```bash
# 1. 原始 WAN + 动作投影图/射线 + VACE
NUM_PROCESSES=16 bash scripts/train_vace.sh

# 2. 原始 WAN + joint14 向量 + AdaLN 与 cross-attention
NUM_PROCESSES=16 bash scripts/train_vector.sh

# 3. 原始 WAN + PhysCanvas 风格网格动作图
NUM_PROCESSES=16 bash scripts/train_physcanvas.sh
```

每条命令各自启动一个训练任务，共用一台机器时请顺序执行。默认配置：**100,000 次 optimizer update，每 5,000 次保存，随后验证 500 个分布式 batch**，BF16、学习率 1e-5、梯度累积 1、ZeRO-2。VACE 更新 DiT 和 VACE；向量/网格更新 DiT 及其动作投影层。脚本不设置集群墙钟时限，由外部调度器配置。

**Optimizer CPU/NVMe offload 默认关闭，并在启动时明确拒绝启用。** 可用 `configs/distributed/zero{0,1,2}.yaml` 选择 ZeRO stage。ZeRO-0 复制优化器状态，ZeRO-2 能运行不意味着 ZeRO-0 也能容纳。

```bash
NUM_PROCESSES=8 ACCELERATE_CONFIG=configs/distributed/zero1.yaml \
  bash scripts/train_vector.sh --set model.action_mode=cross_attn

# 独立短测标签；仍然需要加载完整模型。
NUM_PROCESSES=16 bash scripts/train_vace.sh \
  --set training.max_steps=20 --set training.save_final_checkpoint=false \
  --set output.task=vace-smoke-20steps
```

多机通过 `NUM_MACHINES`、总卡数 `NUM_PROCESSES`、`MACHINE_RANK`、`MASTER_ADDR`、`MASTER_PORT` 配置。每台机器使用正确 rank，并设置相同 `WOVR_RUN_TIMESTAMP`，避免输出到不同目录；所有节点的资产与缓存路径必须一致。

## 参数与变体

YAML 的 `extends` 相对于配置文件解析；重复 `--config` 按顺序覆盖。`--set dotted.key=value` 使用 YAML 值，未知参数直接报错。每个主要参数的含义和候选值写在 `configs/base.yaml` 的注释中。

| 参数/配置 | 含义 |
|---|---|
| `model.action_mode` | `cross_attn` / `adaln` / `cross_attn_and_adaln` |
| `integrated_aug.yaml` | 稀疏历史 + 随机时间步长 + 概率性球形历史模糊 |
| `sparse_history.yaml` | 固定稀疏历史 |
| `temporal_stride.yaml` | 随机时间步长 |
| `sphere_blur.yaml` | 历史帧球形模糊 |
| `single_head.yaml` | 只使用头部相机，调整 VACE view embedding |
| `context1.yaml` | 1 帧历史 |
| `vace_action_only.yaml` | 48 通道：编码动作图 |
| `vace_rawray.yaml` | 54 通道：动作图 + 6 通道射线 |
| `vace_stack4.yaml` | 72 通道：动作图 + stack4 相机射线 |
| `vace_vae_rays.yaml` | 144 通道：分别 VAE 编码动作与射线图 |
| `no_view_embedding.yaml` | 关闭 VACE 相机视角 embedding |

变体文件位于 `configs/variants/`。例如：

```bash
bash scripts/train.sh --config configs/train/robotwin_vace.yaml \
  --config configs/variants/single_head.yaml --dry-run
```

网格 renderer 当前支持三相机流程；单相机变体用于 VACE/向量分支。涉及缓存的变体修改后应重新生成缓存。不同条件结构的 checkpoint 不能互相加载。

## 输出、权重与验证

```text
$WOVR_ASSET_ROOT/outputs/YYYYMMDD-HHMMSS-task/
├── config.json
├── source_snapshot.tar.gz
├── logs/rank-*.log
├── tensorboard/
├── checkpoints/
└── loss_metrics.jsonl
```

`config.json` 保存解析后的运行参数；代码和配置快照随运行保存；所有训练权重统一放入 `checkpoints/`。正常遍历数据，**不再采用每 epoch 501 步截断**。`max_steps` 按 optimizer update 计数，梯度累积时也不改成 microbatch 数。

每轮验证重新 shuffle。16 rank、每 rank batch size 1 时，500 个分布式 batch 对应 8,000 次样本抽取，不是 500 条不同 episode。保存与验证阶段 train loss 暂停更新是正常现象。

通过 `model.checkpoint` 加载的是模型参数初始化，**不是完整恢复 optimizer、scheduler 或 RNG 状态**。例如 `checkpoints/step-00005000.safetensors` 必须搭配原来的方法和结构。VAE、tokenizer、文本编码器仍由外部原始资产提供。

## Loss 指标与性能分析

```bash
tensorboard --logdir "$WOVR_ASSET_ROOT/outputs" --port 6006
```

- `Loss/train_interval`：整个日志区间、跨 rank、按样本加权的平均 loss，推荐作为主比较指标。
- `MSE/train_interval_unweighted`：未乘 diffusion timestep 权重的 MSE。
- `Timestep/train/...`：10 个 timestep 分桶，各自记录加权 loss、未加权 MSE 和计数。判断动作条件的影响前，应比较相近 timestep。
- `Loss/train_step`：当前步抽样，不是区间均值。

前 10 次更新每次记录；16 rank、累积 1 时，step 50 区间覆盖 11–50，共 640 样本；后续每 50 步为 800 样本。每个 rank 的扩散噪声和 timestep 使用独立且可复现的随机流，验证和 dataloader 随机抽样不会推进训练流。绘图可排除前 10 步，并明确说明使用的平滑窗口。

`--set profiling.enabled=true` 会写入 CUDA event/主机阶段计时、显存、`timings.jsonl` 与 `profile_summary.json`；默认统计剔除前 5 个预热步。`profiling.encoder_detail=true` 细分 VAE/文本阶段，`profiling.trace_step=N` 导出一次 rank-0 trace。计时引入同步开销，速度比较需保持相同设置。关闭 profiling 的生产日志不能被解释为实测逐 rank 的编码调用次数。

## 推理

使用匹配的训练权重，在一个验证样本上生成动作条件视频片段：

```bash
bash scripts/infer.sh --config configs/train/adjust_bottle_cached_100k.yaml \
  --set model.checkpoint=/absolute/path/to/checkpoints/step-00005000.safetensors \
  --index 0 --steps 50 --seed 42 --output /your/results/vace-chunk.mp4
```

向量或网格模型使用各自 YAML。推理入口读取原始验证观测和动作，关闭训练缓存，并在 VAE 编码前替换未来 RGB 目标帧。**训练 latent 包含未来监督目标，禁止直接用于生成**。未来动作作为控制条件。输出为一个 chunk，不自动进行闭环 rollout；下游策略控制与评估不在本仓库范围内。默认 `--cfg-scale=1`。

## 代码结构与测试

```text
configs/             # 基础、方法、增强、分布式配置
scripts/             # 三个训练入口，以及下载/缓存/验证/推理
wovrplus/models/     # WAN、VACE、向量条件、网格 patch embedding、VAE/T5
wovrplus/data/       # RoboTwin、投影、FK 网格 renderer、缓存
wovrplus/pipelines/  # 各方法的输入处理和生成
wovrplus/training/   # 优化循环、独立随机流、区间指标、性能分析
tests/               # 小模型前向/梯度、缓存、随机流、训练循环测试
docs/                # 资产、来源、发布验证
```

```bash
python -m unittest discover -s tests -v
# 或安装 test extra 后：
python -m pytest tests -q
```

测试覆盖动作注入模式与参考实现的前向/梯度一致性、网格相加注入、VACE 变体、小样本缓存、rank 随机流隔离、指标计数、offload 拒绝以及超过 501 步的训练循环。小模型测试不能替代完整规模显存与性能验证。本次实际执行内容见 [docs/validation.md](docs/validation.md)。

## 范围、许可与致谢

本仓库聚焦动作条件世界模型训练，不包含无关文生图/文生视频训练样例、云凭证、内部调度脚本、实验权重或完整策略评估系统。数据、缓存、大权重和输出均应位于 Git 仓库之外。

代码为 **Apache-2.0**，公开 adjust_bottle 数据为 **CC BY 4.0**。感谢 [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)、[WAN](https://github.com/Wan-Video/Wan2.2)、[RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin)、[RLinf](https://github.com/RLinf/RLinf)、[PhysCanvas](https://github.com/OpenPhoenix-Team/PhysCanvas)。第三方声明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。文档组织参考 PhysCanvas，本文所述实现范围与验证结果仅对应本仓库。

```bibtex
@misc{youmo4452026wovrplustraining,
  author = {youmo445},
  title = {WoVR+ Action-Conditioned World Model Training},
  year = {2026},
  url = {https://github.com/youmo445/WoVR_Plus_wm_training}
}
```
