# WoVR+ 世界模型训练

**面向机器人动作条件世界模型的 WAN 2.2 训练代码。**

[English](README.md) · [公开数据集](https://huggingface.co/datasets/jzndd/RoboTwin_adjust_bottle_For_ACWM_Training) · [WoVR+ 项目](https://youmo445.github.io/WoVR_Plus.github.io/) · [Apache-2.0](LICENSE)

本仓库在相同 RoboTwin 观测上提供三类动作表征与注入方式：Raymap + VACE、MotionImage + Patch Addition、vector + AdaLN。包含训练、验证集片段生成、可复现 VAE 缓存、分布式优化和 loss 诊断，不依赖同级 DiffSynth-Studio 目录。

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
- [许可与引用](#许可与引用)

## 支持的方法

| 方法 | 动作表征 | 注入方式 | 运行脚本（在线 VAE） |
|---|---|---|---|
| **Raymap + VACE（default）** | 经正向运动学投影的彩色动作图与标定相机射线 | VACE 条件分支，由 WAN DiT 初始化 | `scripts/train_raymap_vace.sh` |
| **MotionImage + Patch Addition**（参考 [PhysCanvas](https://zebinx.github.io/PhysCanvas_HomePage/)） | FK 投影的灰度机器人网格图，经 WAN VAE 编码 | 独立 Conv3d 动作 patch embedding 与视频 patch embedding 相加 | `scripts/train_motionimage.sh` |
| **vector + AdaLN**（参考 [WoVR](https://wovr-rlinf.github.io/)） | 14 维关节目标：左臂 6 + 左夹爪 + 右臂 6 + 右夹爪 | 动作投影后参与 AdaLN 调制；也支持可选 cross-attention 或二者结合 | `scripts/train_vector_adaln.sh` |

三类方法均从 **`Wan-AI/Wan2.2-TI2V-5B` 原始权重**开始。

1. 默认采用三视角、5 帧历史、8 帧未来，每个相机缩放为 256×320。**先在 RGB 空间纵向拼接三路相机，再经过 VAE**，拼接后为 768×320。
2. Raymap 和 MotionImage 需要相机内参和外参。
3. PS：MotionImage + Patch Addition 参考 [PhysCanvas](https://zebinx.github.io/PhysCanvas_HomePage/)，只参考动作表征和注入方式，不涉及触觉等。

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
bash scripts/train_raymap_vace.sh --dry-run
bash scripts/train_vector_adaln.sh --dry-run --set model.action_mode=adaln
bash scripts/train_motionimage.sh --dry-run
```

缓存生成前，可读取原始数据进行预检查：

```bash
bash scripts/train_raymap_vace.sh --validate-only --set data.cache.mode=off
```

预检查核对原始 WAN 的全部 825 个参数键/形状，并读取一个训练和一个验证样本，不分配完整可训练模型。因此它不验证显存是否足够。三个在线入口均可直接执行 `--validate-only`；带 `_cached` 的入口需要先完成对应缓存。

## VAE 缓存

VAE 缓存用于加速训练，省去训练过程中的 VAE 编码耗时，适合快速验证。**预计算 latent 后，训练时无法再重新采样像素级图像增强**；四个缓存版本只是预先固定的增强结果。如果需要在线随机图像增强，请使用不带 `_cached` 的训练入口，实时执行增强和 VAE 编码。

三种对照方法共享 RGB 窗口和增强计划。先构建一份 **VACE 格式 RGB/动作缓存**，再构建向量与网格条件缓存：

```bash
# 遍历全部窗口，每窗口四个固定增强版本；不同 rank 分片并行处理。
NUM_PROCESSES=8 bash scripts/cache_latents.sh \
  --config configs/train/adjust_bottle_raymap_vace_cached_100k.yaml --variants 4

# 读取共享计划中精确 frame_ids 对应的关节向量。
NUM_PROCESSES=8 bash scripts/cache_conditions.sh \
  --config configs/train/adjust_bottle_vector_adaln_cached_100k.yaml

# FK 网格灰度画布 -> VAE motion latent。
NUM_PROCESSES=8 bash scripts/cache_conditions.sh \
  --config configs/train/adjust_bottle_motionimage_cached_100k.yaml
```

全窗口缓存可能需要数小时及显著大于原始压缩数据的存储空间。每个 rank 处理互不重叠的分片，只有完成覆盖核验才写入 `complete` manifest。中断后可用原配置恢复并复用已核验文件，不要同时启动两个写同一目录的 builder。

四个版本固定的是窗口采样和数据增强，**不缓存扩散噪声或 timestep**；这两者仍按 rank、训练更新独立随机采样。缓存训练省去 RGB/动作 VAE 编码；网格路径需要额外预计算 motion latent。VACE 和网格模型仍使用文本编码器；向量配置默认关闭文本。

读取器检查预处理代码指纹、源文件身份、条件签名、张量形状和 dtype。`data.cache.verify_checksum=true` 可额外逐次核验 SHA256。修改路径、预处理代码、视角布局、分辨率、采样或增强后，需要对应的新缓存，通常应指定新的缓存目录重新构建。不要直接复用其他代码版本的内部缓存。

在线和缓存入口见下表。做在线/缓存数值对照时，必须使用相同采样计划。`scripts/validate_latent_cache.py` 用于 Raymap + VACE 在线重放与缓存的一致性检查。

## 三组训练入口

每种方法均提供在线编码与 VAE cache 两个入口；缓存版本需先完成上面的构建步骤。

| 方法 | 在线编码（支持随机图像增强） | VAE cache |
|---|---|---|
| Raymap + VACE | `train_raymap_vace.sh` | `train_raymap_vace_cached.sh` |
| MotionImage + Patch Addition | `train_motionimage.sh` | `train_motionimage_cached.sh` |
| vector + AdaLN | `train_vector_adaln.sh` | `train_vector_adaln_cached.sh` |

```bash
# 不使用 VAE cache：从原始 HDF5 读取，在线增强和编码。
NUM_PROCESSES=16 bash scripts/train_raymap_vace.sh
NUM_PROCESSES=16 bash scripts/train_motionimage.sh
NUM_PROCESSES=16 bash scripts/train_vector_adaln.sh

# 使用 VAE cache：直接读取预计算 latent。
NUM_PROCESSES=16 bash scripts/train_raymap_vace_cached.sh
NUM_PROCESSES=16 bash scripts/train_motionimage_cached.sh
NUM_PROCESSES=16 bash scripts/train_vector_adaln_cached.sh
```

每条命令各自启动一个训练任务，共用一台机器时请顺序执行。默认配置：**100,000 次 optimizer update，每 5,000 次保存，随后验证 500 个分布式 batch**，BF16、学习率 1e-5、梯度累积 1、ZeRO-2。VACE 更新 DiT 和 VACE；向量/网格更新 DiT 及其动作投影层。脚本不设置集群墙钟时限，由外部调度器配置。

可用 `configs/distributed/zero{0,1,2}.yaml` 选择 ZeRO stage。

```bash
NUM_PROCESSES=8 ACCELERATE_CONFIG=configs/distributed/zero1.yaml \
  bash scripts/train_vector_adaln.sh --set model.action_mode=cross_attn

# 独立短测标签；仍然需要加载完整模型。
NUM_PROCESSES=16 bash scripts/train_raymap_vace.sh \
  --set training.max_steps=20 --set training.save_final_checkpoint=false \
  --set output.task=vace-smoke-20steps
```

多机通过 `NUM_MACHINES`、总卡数 `NUM_PROCESSES`、`MACHINE_RANK`、`MASTER_ADDR`、`MASTER_PORT` 配置。每台机器使用正确 rank，并设置相同 `WOVR_RUN_TIMESTAMP`，避免输出到不同目录；所有节点的资产与缓存路径必须一致。

## 参数与变体

YAML 的 `extends` 相对于配置文件解析；重复 `--config` 按顺序覆盖。`--set dotted.key=value` 使用 YAML 值，未知参数直接报错。每个主要参数的含义和候选值写在 `configs/base.yaml` 的注释中。

| 参数/配置 | 含义 |
|---|---|
| `model.action_mode` | `adaln`（默认）/ `cross_attn` / `cross_attn_and_adaln` |
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
bash scripts/train.sh --config configs/train/robotwin_raymap_vace.yaml \
  --config configs/variants/single_head.yaml --dry-run
```

网格 renderer 当前支持三相机流程；单相机变体用于 VACE/向量分支。涉及缓存的变体修改后应重新生成缓存。不同条件结构的 checkpoint 不能互相加载。

旧 Python 导入路径、方法名和配置文件保留兼容入口；新配置使用 `raymap_vace`、`motionimage`、`vector`。加载旧的双注入向量 checkpoint 时，应显式设置 `model.action_mode=cross_attn_and_adaln`。预处理指纹随命名更新，旧缓存需要在新目录重建。

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
bash scripts/infer.sh --config configs/train/adjust_bottle_raymap_vace_cached_100k.yaml \
  --set model.checkpoint=/absolute/path/to/checkpoints/step-00005000.safetensors \
  --index 0 --steps 50 --seed 42 --output /your/results/vace-chunk.mp4
```

向量或网格模型使用各自 YAML。推理入口读取原始验证观测和动作，关闭训练缓存，并在 VAE 编码前替换未来 RGB 目标帧。**训练 latent 包含未来监督目标，禁止直接用于生成**。未来动作作为控制条件。输出为一个 chunk，不自动进行闭环 rollout；下游策略控制与评估不在本仓库范围内。默认 `--cfg-scale=1`。

## 代码结构与测试

```text
configs/             # 基础、方法、增强、分布式配置
scripts/             # 三种方法的在线/缓存入口，以及下载/缓存/验证/推理
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

## 许可与引用

本项目采用 **[Apache-2.0](LICENSE)** 许可。如果本项目对你的研究有帮助，欢迎引用我们的论文：

```bibtex
@article{jiang2026wovr,
  title={WoVR: World Models as Reliable Simulators for Post-Training VLA Policies with RL},
  author={Zhennan Jiang and Shangqing Zhou and Yutong Jiang and Zefang Huang and Mingjie Wei and Yuhui Chen and Tianxing Zhou and Zhen Guo and Hao Lin and Quanlu Zhang and Yu Wang and Haoran Li and Chao Yu and Dongbin Zhao},
  year={2026},
  journal={arXiv preprint arXiv:2602.13977},
}

@article{zhou2026wovrworldmodelreliable,
  title={WoVR+: World Model as Reliable Simulator for Versatile VLA Policy Optimization with RL},
  author={Shangqing Zhou and Zhennan Jiang and Ning Ma and Xin Fu and Yupeng Zheng and Shuai Tian and Huangrui Li and Jiahao Liu and Chao Yu and Zhongpu Xia and Haoran Li and Dongbin Zhao},
  year={2026},
  journal={arXiv preprint arXiv:2610.17922},
  eprint={2610.17922}
}
```
