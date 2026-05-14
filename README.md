# DSCA-ReID Framework

A lightweight PyTorch research framework for **Diffusion-based Semantic Camouflage Attack (DSCA)** in person re-identification.

This repository is designed for **code release, method understanding, and follow-up research extension**. It provides a clean and runnable implementation of the main DSCA module boundaries, including condition extraction, USCC fusion, dual-pathway identity injection, DDIM generation, surrogate-based identity alignment, and ReID-style evaluation metrics.

> **Scope note**  
> This repository does **not** bundle pretrained Stable Diffusion, OpenPose, DeepLabV3, CLIP, FastReID, OSNet, CPT, or other external checkpoints. The default backend is lightweight so that the project can run immediately after installation. To reproduce paper-level experimental numbers, replace the lightweight components with the corresponding pretrained external models and use the official ReID benchmark protocols.

## Main features

- **Unified Semantic Condition Composer (USCC)**
  - pose, foreground mask, and style token adapters
  - reliability-aware temperature-softmax gating
  - edge/skeleton masks and foreground attention priors

- **Dual-pathway identity injection**
  - structure path with ControlNet-style residuals
  - style path with foreground-masked cross-attention
  - per-layer FiLM controller
  - explicit depth scheduling for structure and appearance paths

- **Conditional diffusion generation**
  - DDPM-style training objective
  - deterministic DDIM inference
  - source-anchored latent trajectory

- **Training and evaluation utilities**
  - mixed precision on CUDA
  - cosine learning-rate schedule
  - EMA weights
  - resumable checkpoints
  - ASR, Rank-k, CMC, mAP, PSNR, SSIM, and surrogate-victim discrepancy metrics

## Repository layout

```text
DSCA-ReID-Framework/
├── configs/
│   ├── default.yaml
│   └── lightweight.yaml
├── data/
│   └── README.md
├── docs/
│   ├── REPRODUCTION_NOTE.md
│   └── REPLACEMENT_GUIDE.md
├── scripts/
│   ├── smoke_check.py
│   ├── train_dsca.py
│   ├── infer_dsca.py
│   └── eval_dsca.py
├── src/dsca/
│   ├── data/
│   ├── models/
│   ├── samplers/
│   ├── utils/
│   ├── config.py
│   ├── inferencer.py
│   ├── losses.py
│   ├── metrics.py
│   └── trainer.py
├── tests/
├── CITATION.cff
├── LICENSE
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

## Installation

```bash
git clone https://github.com/<your-name>/DSCA-ReID-Framework.git
cd DSCA-ReID-Framework
pip install -r requirements.txt
pip install -e .
```

For development and tests:

```bash
pip install -r requirements-dev.txt
pytest -q
```

## PyCharm import

1. Open PyCharm.
2. Choose **File → Open** and select the repository root.
3. Select or create a Python interpreter.
4. Install the dependencies with the commands above.
5. If PyCharm does not detect the source directory automatically, right click `src` and choose **Mark Directory as → Sources Root**.
6. Run the smoke check:

```bash
python scripts/smoke_check.py --config configs/lightweight.yaml --out outputs/smoke
```

A successful run confirms that forward propagation, backward propagation, DDIM generation, loss computation, and file output all work.

## Quick start

### 1. Smoke check

```bash
python scripts/smoke_check.py --config configs/lightweight.yaml --out outputs/smoke
```

### 2. Train

Edit `configs/default.yaml` first:

```yaml
data:
  train_dir: /path/to/Market-1501-v15.09.15/bounding_box_train
```

Then run:

```bash
python scripts/train_dsca.py --config configs/default.yaml
```

Resume from a checkpoint:

```bash
python scripts/train_dsca.py \
  --config configs/default.yaml \
  --resume checkpoints/dsca_last.pt
```

### 3. Inference

```bash
python scripts/infer_dsca.py \
  --config configs/default.yaml \
  --source path/to/source.jpg \
  --target path/to/target.jpg \
  --checkpoint checkpoints/dsca_last.pt \
  --use-ema \
  --out outputs/infer/camouflage.png
```

### 4. Evaluation

```bash
python scripts/eval_dsca.py \
  --config configs/default.yaml \
  --checkpoint checkpoints/dsca_last.pt \
  --use-ema \
  --max-pairs 512
```

The evaluation script reports targeted attack and retrieval metrics, including ASR, Rank-10, mAP, PSNR, SSIM, and surrogate-victim discrepancy.

## Data format

The loader supports Market-1501 and DukeMTMC-reID style file names:

```text
/path/to/Market-1501-v15.09.15/
└── bounding_box_train/
    ├── 0001_c1s1_001051_00.jpg
    ├── 0001_c2s1_000301_00.jpg
    └── ...
```

Images with person ID `-1` are ignored.

## Replacement points for paper-level experiments

The default implementation uses lightweight modules so that the repository runs without external checkpoints. For stronger experiments, replace the following components:

| Paper component | Current location | Replacement target |
|---|---|---|
| OpenPose pose heatmaps | `src/dsca/models/extractors.py` | OpenPose v1.7.0 wrapper |
| DeepLabV3 human masks | `src/dsca/models/extractors.py` | DeepLabV3 segmentation wrapper |
| CLIP style features | `src/dsca/models/extractors.py` | CLIP ViT-B/32 wrapper |
| Stable Diffusion U-Net and VAE | `src/dsca/models/diffusion_generator.py` | SD latent U-Net and VAE modules |
| ReID surrogate ensemble | `src/dsca/models/surrogates.py` | FastReID, OSNet, Transformer ReID checkpoints |
| Benchmark evaluation | `scripts/eval_dsca.py` | Full Market, Duke, MSMT, PRCC protocol |

See [`docs/REPLACEMENT_GUIDE.md`](docs/REPLACEMENT_GUIDE.md) for details.

## What is implemented

This version includes:

- deterministic DDIM sampler with strictly decreasing sampling indices
- USCC condition fusion with gate entropy regularization support
- edge/skeleton structure masks and foreground attention priors
- depth-scheduled structure and style paths
- foreground-masked cross-attention for style injection
- per-layer FiLM modulation
- heterogeneous lightweight surrogate ensemble
- identity transfer, visual consistency, diffusion denoising, and gate-diversity losses
- training with AMP, EMA, cosine LR, gradient accumulation, and checkpoint resume
- ASR, CMC/mAP, Rank-k, PSNR, SSIM, and surrogate-victim discrepancy metrics
- unit tests and a CPU-friendly smoke check

## Reproduction note

This repository is suitable for releasing a DSCA-style code framework. It is not a one-click reproduction package for all paper tables. See [`docs/REPRODUCTION_NOTE.md`](docs/REPRODUCTION_NOTE.md) for the precise reproduction boundary.

## Ethics and responsible use

This code is intended for authorized robustness evaluation, red-teaming research, and defense development for person re-identification systems. Do not use it to attack systems without permission.

## Citation

If this repository helps your work, please cite the corresponding paper. A `CITATION.cff` file is included for GitHub citation support.

```bibtex
@article{du2026dsca,
  title   = {Security Enhancement for Person Re-Identification through Diffusion Driven Semantic Attacks},
  author  = {Du, Kaixin and Ma, Bin and Yang, Meihong and Xu, Jian and Li, Xiaolong},
  journal = {IEEE Transactions on Information Forensics and Security},
  year    = {2026},
  note    = {Accepted as a regular paper}
}
```

## License

This repository is released under the MIT License. See [`LICENSE`](LICENSE) for details.
