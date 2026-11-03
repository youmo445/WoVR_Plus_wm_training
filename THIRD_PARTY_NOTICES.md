# Third-party notices

This repository is distributed under Apache-2.0; see [LICENSE](LICENSE). Existing upstream copyright and license notices are retained. New packaging and documentation do not change the licenses of upstream components or externally downloaded assets.

| Component / reference | Origin | Use in this repository |
|---|---|---|
| WAN model, VAE, T5 and pipeline utilities | [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), Apache-2.0; [Wan2.2](https://github.com/Wan-Video/Wan2.2), Apache-2.0 | Focused model/pipeline implementation and original WAN weight format |
| Direct vector conditioning | [diffsynth-studio-rlinf](https://github.com/youmo445/diffsynth-studio-rlinf), derived from DiffSynth-Studio/RLinf | Cross-attention and AdaLN implementation; reference fixtures for numerical/gradient tests |
| RLinf license notice | [RLinf](https://github.com/RLinf/RLinf), Apache-2.0 | Retained in [docs/rlinf-LICENSE](docs/rlinf-LICENSE) |
| Mesh action representation/injection concept | [PhysCanvas](https://github.com/OpenPhoenix-Team/PhysCanvas), Apache-2.0 | Adapted grayscale FK mesh projection and additive patch embedding; not its full multimodal system |
| Robot simulation and embodiment assets | [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin), MIT; [asset repository](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0) | Downloaded separately; no robot meshes or upstream simulation package bundled |
| Downstream rollout dataset | [jzndd/RoboTwin_adjust_bottle_For_ACWM_Training](https://huggingface.co/datasets/jzndd/RoboTwin_adjust_bottle_For_ACWM_Training), CC BY 4.0 | External HDF5 training/validation data |

Dependencies installed through pip retain their respective licenses. Model-weight terms are those of the original model repository. This code release does not distribute pretrained experiment checkpoints, training data, or third-party asset archives in Git.

Please cite the upstream methods when using them in scientific work, in addition to this release and the dataset. The PhysCanvas paper is [Physical Canvas: A Vision Tactile World Model For Robot Policy Evaluation And Learning](https://arxiv.org/abs/2610.28613). The RoboTwin project provides its current citation in its upstream README.
