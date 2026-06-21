"""Collator: construye mixture a partir de target + residual."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torchaudio

from src.data.slakh.sample_builder import RawSample
from src.data.processing.waveform_utils import peak_normalize


@dataclass
class CollatorConfig:
    snr_mean: float = -2.5
    snr_std: float = 1.5
    snr_min: float = -5.0
    snr_max: float = 5.0
    remix: bool = True


class GuidedSeparationCollator:
    def __init__(
        self,
        mode: str = "train",
        config: CollatorConfig | None = None,
    ) -> None:
        self.mode = mode
        self.config = config or CollatorConfig()

    def __call__(self, batch: List[RawSample]) -> Dict[str, torch.Tensor]:
        return self.collate(batch)

    def collate(self, batch: List[RawSample]) -> Dict[str, torch.Tensor]:
        target = torch.stack([item.target for item in batch])
        reference = torch.stack([item.reference for item in batch])
        residual = torch.stack([item.residual for item in batch])

        if self.mode == "train" and self.config.remix:
            snr = torch.normal(
                mean=self.config.snr_mean,
                std=self.config.snr_std,
                size=(target.shape[0],),
            )
            snr = torch.clamp(snr, min=self.config.snr_min, max=self.config.snr_max)
            mixture = torchaudio.functional.add_noise(
                waveform=target,
                noise=residual,
                snr=snr,
            )
        else:
            mixture = target + residual

        max_mix = mixture.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
        max_tar = target.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
        max_ref = reference.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)

        mixture = mixture / max_mix
        target = target / max_tar
        reference = reference / max_ref

        return {
            "mixture": mixture,
            "reference": reference,
            "target": target,
        }
