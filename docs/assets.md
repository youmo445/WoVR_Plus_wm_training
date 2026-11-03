# External assets and reproducibility

All large assets live under `WOVR_ASSET_ROOT`, outside the Git checkout. Paths in YAML are expanded before construction, including lists of dataset directories. The default when the variable is unset is the sibling directory `../wovr-assets`. Prefer an explicitly set absolute path across machines.

## Sources

| Asset | Public source | Destination |
|---|---|---|
| Five adjust_bottle rollout groups | [jzndd/RoboTwin_adjust_bottle_For_ACWM_Training](https://huggingface.co/datasets/jzndd/RoboTwin_adjust_bottle_For_ACWM_Training) | `datasets/adjust_bottle` |
| Original WAN model, VAE, T5, tokenizer | [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) | `models/Wan2.2-TI2V-5B` |
| ALOHA/AgileX robot config, URDF, meshes | [TianxingChen/RoboTwin2.0](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0), `embodiments.zip` | `robot/aloha-agilex` |

`scripts/download_assets.py` resolves the requested revision to a commit and records it in `{data,wan,robot}-download-provenance.json` outside the repo. Use `--revision` for a fixed revision of one source. Keep these JSON files with experiment records. Upstream licenses remain applicable; the downloader does not relicense assets.

The robot downloader selects only the `aloha-agilex` subtree from the official archive. For manual installation, keep `config.yml` together with the URDF, SRDF and mesh tree, preserving all relative paths. Do not copy a lone URDF or replace it with a visually similar robot: FK geometry and camera-space action conditions would change. The mesh renderer expects the RoboTwin left/right link6/7/8 COLLADA geometry and rejects unsupported transforms rather than silently approximating it.

## Data integrity and paths

Run `python scripts/verify_dataset.py /path/to/adjust_bottle` after download. It streams each HDF5 file and compares its SHA256 against the published manifest. The dataset's HDF5 arrays were preserved from the source; only internal reference-path attributes were normalized. See the dataset card for scene splitting and generation provenance.

Cached latent manifests include preprocessing code fingerprints and source identities (path, size and mtime), plus model-asset fingerprints. Moving a cache and changing YAML strings is not a supported way to bypass provenance. Build a matching cache at its intended final location. Condition caches retain the frame IDs/camera order from the shared RGB plan. Incomplete manifests must not be used for normal training.

## Storage planning

The raw public HDF5 files occupy about 5.29 GiB. Model downloads are much larger, and full-window four-variant caches can grow to hundreds of GiB depending on geometry/representation. Reserve space for original weights, temporary downloads, cache files, and all retained checkpoints. The downloader does not delete training data or checkpoints automatically.

## 中文要点

资产均放在仓库外，通过 `WOVR_ASSET_ROOT` 定位；缓存生成与训练保持相同路径。下载器记录上游 commit，机器人需完整保留 config/URDF/mesh 相对路径。运行 `verify_dataset.py` 核验 500 个 HDF5 的 SHA256。缓存包含代码与源文件身份，修改路径、预处理或资产后应重新生成匹配缓存，不应跳过校验。数据约 5.29 GiB，权重和全窗口四版本缓存需要更多空间。
