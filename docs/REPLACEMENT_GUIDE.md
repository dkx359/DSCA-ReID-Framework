# Replacement Guide

This guide explains where to connect external pretrained modules when moving from the lightweight framework to paper-level experiments.

## 1. Condition extractors

File:

```text
src/dsca/models/extractors.py
```

Expected output:

```python
DSCAConditions(
    pose: Tensor[B, C_pose, H, W],
    mask: Tensor[B, 1, H, W],
    style: Tensor[B, style_dim],
)
```

Recommended replacements:

- OpenPose v1.7.0 for pose heatmaps
- DeepLabV3 for human foreground segmentation
- CLIP ViT-B/32 for foreground style features

Keep tensor shapes and value ranges consistent with the lightweight extractor.

## 2. Diffusion backbone

File:

```text
src/dsca/models/diffusion_generator.py
```

Current modules:

- `LatentEncoder`
- `LatentDecoder`
- `ConditionalDenoisingUNet`

Recommended replacements:

- Stable Diffusion VAE encoder
- Stable Diffusion VAE decoder
- Stable Diffusion U-Net

The existing `USCCOutput` and dual-path injection interfaces can be reused as adapter points.

## 3. Surrogate ensemble

File:

```text
src/dsca/models/surrogates.py
```

Current implementation:

- lightweight CNN-style extractor
- lightweight Transformer-style extractor
- ensemble wrapper with normalized feature output

Recommended replacements:

- FastReID ResNet-50 or ResNet-100 model
- OSNet model
- Transformer-based ReID model

The ensemble should return a list of feature tensors with shape:

```python
Tensor[B, D]
```

For transferability experiments, keep the surrogate set architecturally heterogeneous.

## 4. Evaluation protocol

File:

```text
scripts/eval_dsca.py
```

The script currently exposes the core metric interfaces. For benchmark-level reporting, extend it with:

- query/gallery splits
- camera-aware filtering
- same-camera removal when required by the benchmark
- Rank-1, Rank-5, Rank-10, mAP, and mINP
- target identity sampling protocol
- cross-dataset evaluation
- defense evaluation hooks

## 5. Practical checklist

Before running a paper-level experiment, check:

- image size matches the pretrained diffusion and ReID modules
- all feature extractors are frozen unless intentionally fine-tuned
- source and target identities are disjoint where required
- victim models are excluded from the surrogate ensemble
- generated images are saved with consistent preprocessing and normalization
- ASR is computed under the same target retrieval definition used in the paper
