# Reproduction Note

This repository provides a runnable and extensible DSCA-style research framework. It intentionally separates the method logic from large external dependencies and pretrained checkpoints.

## What can be run immediately

The following components work without external pretrained models:

- lightweight condition extraction
- USCC fusion
- dual-pathway injection
- DDIM generation
- surrogate identity-alignment loss
- visual and diffusion losses
- smoke check
- unit tests
- training and inference scripts
- ReID-style metric computation

## What is required for paper-level numbers

To reproduce paper-level experimental results, the lightweight modules must be replaced with real pretrained components:

1. Stable Diffusion latent U-Net and VAE
2. OpenPose v1.7.0 pose extractor
3. DeepLabV3 human segmentation model
4. CLIP ViT-B/32 style encoder
5. pretrained ReID surrogate models, such as FastReID, OSNet, and a Transformer-based ReID model
6. official Market-1501, DukeMTMC-reID, MSMT17, and PRCC protocols where applicable
7. full victim-model evaluation and defense-comparison protocols

## Recommended wording for public release

Recommended:

- lightweight PyTorch research framework
- reference implementation of the DSCA module design
- extensible implementation for ReID robustness research

Avoid claiming:

- one-click reproduction of all paper tables
- official pretrained model release
- immediate reproduction of paper-level ASR without external checkpoints

## Why this separation is useful

The modular design makes it easier to inspect, debug, and replace each paper component independently. This is particularly useful for follow-up research on semantic attacks, ReID robustness, diffusion-based data generation, and defense evaluation.
