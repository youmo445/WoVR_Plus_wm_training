# WoVR+ World Model Training

**A focused WAN 2.2 training codebase for action-conditioned robot world models.**

[中文文档](README_zh.md) · [Public dataset](https://huggingface.co/datasets/jzndd/RoboTwin_adjust_bottle_For_ACWM_Training) · [WoVR+ project](https://youmo445.github.io/WoVR_Plus.github.io/) · [Apache-2.0](LICENSE)

Train three action-conditioning families on the same RoboTwin observations: Raymap + VACE, MotionImage + Patch Addition, and vector + AdaLN. The package includes training, held-out chunk generation, deterministic VAE caching, distributed optimization, and loss diagnostics. It runs independently of a neighboring DiffSynth-Studio checkout.

## Contents

- [Supported methods](#supported-methods)
- [Installation](#installation)
- [Download data and assets](#download-data-and-assets)
- [Quick start](#quick-start)
- [VAE caches](#vae-caches)
- [Three training recipes](#three-training-recipes)
- [Configuration and variants](#configuration-and-variants)
- [Outputs, checkpoints and evaluation](#outputs-checkpoints-and-evaluation)
- [Loss metrics and profiling](#loss-metrics-and-profiling)
- [Inference](#inference)
- [Project structure and tests](#project-structure-and-tests)
- [License and citation](#license-and-citation)

## Supported methods

| Method | Action representation | Injection | Entry script (online VAE) |
|---|---|---|---|
| **Raymap + VACE (default)** | FK-projected colored action images and calibrated camera rays | VACE conditioning branch initialized from WAN DiT | `scripts/train_raymap_vace.sh` |
| **MotionImage + Patch Addition** (inspired by [PhysCanvas](https://zebinx.github.io/PhysCanvas_HomePage/)) | FK-projected grayscale robot meshes encoded by WAN VAE | Independent Conv3d action patch embeddings added to video patch embeddings | `scripts/train_motionimage.sh` |
| **vector + AdaLN** (inspired by [WoVR](https://wovr-rlinf.github.io/)) | 14D joint targets: left arm 6, left gripper, right arm 6, right gripper | Projected actions modulate AdaLN; optional cross-attention or both remain available | `scripts/train_vector_adaln.sh` |

All three start from **original `Wan-AI/Wan2.2-TI2V-5B` weights**.

1. Defaults are three views, five context frames, and eight future frames, with each camera resized to 256×320. **The three views are stacked vertically in RGB space before VAE encoding**, producing a 768×320 canvas.
2. Raymap and MotionImage require camera intrinsics and extrinsics.
3. PS: MotionImage + Patch Addition draws on [PhysCanvas](https://zebinx.github.io/PhysCanvas_HomePage/) only for its action representation and injection, without tactile components.

## Installation

Use Linux and Python 3.10 or newer. Install a PyTorch build matching your accelerator/driver first; the package requires PyTorch ≥2.4. BF16-capable accelerators and substantial host memory are needed for full WAN 5B fine-tuning. The distributed recipes use DeepSpeed. Optional FlashAttention is detected when installed; the attention implementation otherwise uses PyTorch SDPA. Do not install an incompatible CUDA wheel over a vendor accelerator environment.

```bash
git clone https://github.com/youmo445/WoVR_Plus_wm_training.git
cd WoVR_Plus_wm_training
python -m venv .venv
source .venv/bin/activate
# Install the appropriate torch/torchvision build for your machine first.
pip install -e '.[distributed,test]'
```


## Download data and assets

Choose a storage directory **outside the checkout**. Set the same absolute value for cache construction, training, and inference; caches record asset identities and source paths.

```bash
export WOVR_ASSET_ROOT=/your/storage/wovr-assets
python scripts/download_assets.py data
python scripts/download_assets.py wan
python scripts/download_assets.py robot
python scripts/verify_dataset.py "$WOVR_ASSET_ROOT/datasets/adjust_bottle"
```

The downloader resolves Hub revisions to commits and writes per-asset provenance JSON. It uses normal Hugging Face authentication if your environment requires it; no credentials are stored in this repository. Use `--revision COMMIT` separately for each asset to pin a download.

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

The public dataset contains **500 episodes / 121,263 timesteps / 5.68 GB**. `adjust_bottle` is ODE rollout; `_1`–`_4` are SDE rollouts. Author-provided settings are chunkwise-noise, level 0.2, and five policy sampling steps; the [dataset card](https://huggingface.co/datasets/jzndd/RoboTwin_adjust_bottle_For_ACWM_Training) distinguishes verified file attributes from author metadata. Images are simulator observations, and failures are retained.

The supplied split uses episode IDs 0–79 in every group for training and 80–99 for validation: **400/100 rollouts, 80/20 scenes**. Equal IDs share scene seeds across rollout groups. Do not randomly split files across these related scenes.

Robot assets are downloaded separately from the official RoboTwin `embodiments.zip`, extracting only `aloha-agilex`. WAN weights, robot assets, and the dataset retain their respective upstream licenses. See [asset details](docs/assets.md) for manual setup and versioning.

## Quick start

Inspect a configuration without opening weights or data:

```bash
bash scripts/train_raymap_vace.sh --dry-run
bash scripts/train_vector_adaln.sh --dry-run --set model.action_mode=adaln
bash scripts/train_motionimage.sh --dry-run
```

For a raw-data preflight before building caches:

```bash
bash scripts/train_raymap_vace.sh --validate-only --set data.cache.mode=off
```

Preflight checks all 825 original WAN tensor keys/shapes and reads one training and validation sample. It does not allocate the full trainable model or prove GPU memory feasibility. All three online entries support `--validate-only`; `_cached` entries require their corresponding caches first.

## VAE caches

VAE caching accelerates training by avoiding online VAE encoding and is useful for quick validation. **Precomputed latents cannot receive newly sampled pixel-space image augmentations during training**; four cache variants only preserve four fixed augmentation outcomes. Use the entries without `_cached` for online random image augmentation followed by VAE encoding.

The three comparison recipes share a deterministic plan of RGB windows and augmentations. Build **one VACE-format RGB/action cache** first, then derive the two alternative action-condition caches:

```bash
# All windows; four fixed augmentation variants per window; ranks shard the work.
NUM_PROCESSES=8 bash scripts/cache_latents.sh \
  --config configs/train/adjust_bottle_raymap_vace_cached_100k.yaml --variants 4

# Joint vectors at the exact recorded frame IDs.
NUM_PROCESSES=8 bash scripts/cache_conditions.sh \
  --config configs/train/adjust_bottle_vector_adaln_cached_100k.yaml

# Grayscale FK mesh canvases -> VAE motion latents.
NUM_PROCESSES=8 bash scripts/cache_conditions.sh \
  --config configs/train/adjust_bottle_motionimage_cached_100k.yaml
```

Cache construction can take hours and requires much more disk than the compressed HDF5 dataset. Each rank processes a disjoint subset. A manifest is marked `complete` only after coverage checks. Resume the same build with the same configuration to reuse verified files; do not run duplicate builders against one cache root.

Four versions fix four sampling/augmentation outcomes per window. **They never cache diffusion noise or timesteps**: training draws those independently per rank and step. Cached training avoids RGB/action VAE encoding; vector conditions are indexed from joint targets, while mesh conditions require their separate precomputation. Text encoding remains active for VACE and mesh models; it is disabled by default for the vector recipe.

Cache readers verify the preprocessing fingerprint, source identities, condition signature, tensor shape and dtype. Optional `data.cache.verify_checksum=true` checks each tensor file on read. Changing the source path, preprocessing code, view layout, resolution, sampling, or augmentation requires a matching cache, usually a new root and rebuild. Do not reuse unrelated private caches with this public checkout.

Online and cached entries are listed below. Use equivalent sampling plans for numerical comparisons. `scripts/validate_latent_cache.py` compares Raymap + VACE online replay against cached latents.

## Three training recipes

Each method has online and cached entries. Complete the cache-building steps before using a cached entry.

| Method | Online encoding (random image augmentation) | VAE cache |
|---|---|---|
| Raymap + VACE | `train_raymap_vace.sh` | `train_raymap_vace_cached.sh` |
| MotionImage + Patch Addition | `train_motionimage.sh` | `train_motionimage_cached.sh` |
| vector + AdaLN | `train_vector_adaln.sh` | `train_vector_adaln_cached.sh` |

```bash
# No VAE cache: read raw HDF5, augment and encode online.
NUM_PROCESSES=16 bash scripts/train_raymap_vace.sh
NUM_PROCESSES=16 bash scripts/train_motionimage.sh
NUM_PROCESSES=16 bash scripts/train_vector_adaln.sh

# VAE cache: read precomputed latents.
NUM_PROCESSES=16 bash scripts/train_raymap_vace_cached.sh
NUM_PROCESSES=16 bash scripts/train_motionimage_cached.sh
NUM_PROCESSES=16 bash scripts/train_vector_adaln_cached.sh
```

Each command starts a separate job; run them sequentially if they share one machine. Defaults: **100,000 optimizer updates, checkpoint every 5,000 updates, 500 distributed validation batches after each checkpoint**, BF16, learning rate 1e-5, gradient accumulation 1, and ZeRO-2. VACE trains DiT and VACE; vector/mesh recipes train DiT including their new action projections. These configurations do not impose a scheduler wall-clock limit: set any cluster time limit in your own launcher.

Use `configs/distributed/zero{0,1,2}.yaml` to select the ZeRO stage.

```bash
NUM_PROCESSES=8 ACCELERATE_CONFIG=configs/distributed/zero1.yaml \
  bash scripts/train_vector_adaln.sh --set model.action_mode=cross_attn

# Disposable short run: use a distinct output label; this still loads the full model.
NUM_PROCESSES=16 bash scripts/train_raymap_vace.sh \
  --set training.max_steps=20 --set training.save_final_checkpoint=false \
  --set output.task=vace-smoke-20steps
```

Multi-node launch is supported through `NUM_MACHINES`, total `NUM_PROCESSES`, `MACHINE_RANK`, `MASTER_ADDR`, and `MASTER_PORT`. Launch the same command on each node with the proper rank and a shared `WOVR_RUN_TIMESTAMP` so all ranks use the same output directory. Asset/cache paths must be shared and identical.

## Configuration and variants

YAML inheritance is relative to the YAML file; repeated `--config` overlays merge in order. `--set dotted.key=value` uses YAML values and rejects unknown keys. Parameter captions and choices are documented in `configs/base.yaml`.

| Setting / overlay | Options or effect |
|---|---|
| `model.action_mode` | `adaln` (default), `cross_attn`, `cross_attn_and_adaln` |
| `configs/variants/integrated_aug.yaml` | Sparse history + random temporal stride + probabilistic spherical context blur |
| `sparse_history.yaml` | Fixed sparse history |
| `temporal_stride.yaml` | Random temporal stride |
| `sphere_blur.yaml` | Context spherical blur |
| `single_head.yaml` | Head camera only; VACE view embedding adjusted |
| `context1.yaml` | One context frame instead of five |
| `vace_action_only.yaml` | 48-channel encoded action image |
| `vace_rawray.yaml` | 54 channels: action image + six ray channels |
| `vace_stack4.yaml` | 72 channels: action image + stacked camera rays |
| `vace_vae_rays.yaml` | 144 channels: VAE-encoded action and ray images |
| `no_view_embedding.yaml` | Disable VACE camera-view embeddings |

Example: `bash scripts/train.sh --config configs/train/robotwin_raymap_vace.yaml --config configs/variants/single_head.yaml --dry-run`. The mesh renderer currently supports the three-camera recipe; single-camera alternatives are for VACE/vector paths. Changing cache-related settings requires regeneration. A checkpoint from one condition architecture cannot be loaded into a different architecture.

Legacy Python imports, family names and config paths remain as compatibility entries; new configs use `raymap_vace`, `motionimage`, and `vector`. To load an older dual-injection vector checkpoint, explicitly set `model.action_mode=cross_attn_and_adaln`. Renaming updates preprocessing fingerprints, so rebuild old caches under a new root.

## Outputs, checkpoints and evaluation

```text
$WOVR_ASSET_ROOT/outputs/YYYYMMDD-HHMMSS-task/
├── config.json                 # resolved runtime arguments
├── source_snapshot.tar.gz      # source/config snapshot
├── logs/rank-*.log
├── tensorboard/
├── checkpoints/                # all trained weights, step-based names
└── loss_metrics.jsonl          # interval and timestep-bin records
```

Data are traversed normally; **there is no 501-step epoch truncation**. `max_steps` counts optimizer updates, including under gradient accumulation. Validation is reshuffled each round. At 16 ranks, 500 distributed batches cover 8,000 sample draws when each rank has batch size one; this is not 500 distinct episodes. Train-loss logging pauses during checkpoint serialization and evaluation.

Checkpoints contain trained-family model weights. Loading `model.checkpoint=/absolute/path/checkpoints/step....safetensors` is a **weights initialization**, not exact optimizer/scheduler/RNG-state resumption. Reproduce the original family and architecture when loading. The base VAE/tokenizer/text encoder remain external assets.

## Loss metrics and profiling

```bash
tensorboard --logdir "$WOVR_ASSET_ROOT/outputs" --port 6006
```

- `Loss/train_interval`: sample-weighted mean over the **whole logging interval across ranks**. Use this for comparison.
- `MSE/train_interval_unweighted`: the corresponding MSE before diffusion timestep weighting.
- `Timestep/train/...`: ten timestep bins, each with weighted loss, unweighted MSE and count. Compare like-for-like bins before attributing a difference to action conditioning.
- `Loss/train_step`: a current-step sample, not an interval mean.

The first ten updates are logged individually. With 16 ranks and accumulation 1, the step-50 interval covers updates 11–50 (640 samples); subsequent 50-step intervals contain 800 samples. Each rank has separate deterministic diffusion noise/timestep streams; validation and dataloader draws do not advance the training stream. For plots, excluding the first ten warmup updates is often useful; state any smoothing window explicitly.

Enable timing with `--set profiling.enabled=true`. This writes CUDA-event/host stage timings, memory measurements, `timings.jsonl`, and `profile_summary.json`; the default warmup exclusion is five steps. `profiling.encoder_detail=true` expands VAE/text spans. `profiling.trace_step=N` exports one rank-0 trace. Profiling adds synchronization overhead; compare runs with the same instrumentation. With profiling disabled, training logs do not establish measured per-rank encoder-call counts.

## Inference

Generate one held-out action-conditioned chunk from a matching trained checkpoint:

```bash
bash scripts/infer.sh --config configs/train/adjust_bottle_raymap_vace_cached_100k.yaml \
  --set model.checkpoint=/absolute/path/to/checkpoints/step-00005000.safetensors \
  --index 0 --steps 50 --seed 42 --output /your/results/vace-chunk.mp4
```

Use the vector or mesh YAML for its checkpoint. The inference entry point reads raw held-out observations and actions, switches off training caches, and replaces future RGB target frames before VAE encoding. **Training latent caches contain future targets and are forbidden for generation.** Future actions are the conditioning input. The output is one chunk, not an automatically closed-loop rollout; downstream policy control/evaluation is outside this repository. `--cfg-scale=1` is the default.

## Project structure and tests

```text
configs/             # portable base, method, augmentation and distributed YAMLs
scripts/             # online/cached entries for three methods, download/cache/verification/inference
wovrplus/models/     # WAN, VACE, vector conditioning, mesh patch embedding, VAE/T5
wovrplus/data/       # RoboTwin, projection, FK mesh renderer, cache builders/readers
wovrplus/pipelines/  # method-specific input preparation and generation
wovrplus/training/   # optimizer loop, RNG streams, interval metrics and profiling
tests/               # small model forward/gradient, cache, RNG and training-loop tests
docs/                # assets, provenance and release validation
```

```bash
python -m unittest discover -s tests -v
# Alternatively, after installing the test extra:
python -m pytest tests -q
```

Tests include action-mode output/gradient equivalence against reference fixtures, additive mesh injection, VACE variants, cache equivalence with small fixtures, independent rank RNG, distributed metric accounting, optimizer-offload rejection, and training beyond 501 updates. They are not a substitute for full-scale accelerator testing. See [release validation](docs/validation.md) for the checks actually run on this release.

## License and citation

This project is licensed under **[Apache-2.0](LICENSE)**. If you find it useful for your research, please consider citing our papers:

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
