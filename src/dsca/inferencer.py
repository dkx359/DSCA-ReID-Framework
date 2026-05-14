from __future__ import annotations

import torch
from torch import Tensor

from dsca.models.diffusion_generator import DSCAGenerator
from dsca.models.extractors import ConditionExtractorBase


class DSCAInferencer:
    """Online zero-query camouflage generation (paper Sec. III-E, Algorithm 2 Part 2).

    A single non-iterative pass: extract target conditions, then run the
    deterministic DDIM sampler. No victim interaction occurs here.
    """

    def __init__(
        self,
        generator: DSCAGenerator,
        extractor: ConditionExtractorBase,
        device: torch.device,
    ) -> None:
        self.generator = generator.to(device).eval()
        self.extractor = extractor.to(device).eval()
        self.device = device

    @torch.no_grad()
    def generate(
        self,
        source: Tensor,
        target: Tensor,
        ddim_steps: int = 10,
        temperature: float = 1.0,
        eta: float = 0.0,
    ) -> Tensor:
        source = source.to(self.device)
        target = target.to(self.device)
        conditions = self.extractor(target)
        return self.generator.generate(
            source, conditions, ddim_steps=ddim_steps, temperature=temperature, eta=eta
        )
