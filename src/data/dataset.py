"""GuidedSeparationDataset: dataset minimo para guided MSS."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, TypedDict

import torch
from torch.utils.data import Dataset

from src.data.audio_loader import AudioLoader


class Triplet(TypedDict):
    mixture: str
    reference: str
    target: str


class Sample(TypedDict):
    mixture: torch.Tensor
    reference: torch.Tensor
    target: torch.Tensor


class GuidedSeparationDataset(Dataset):
    def __init__(
        self,
        triplets: List[Triplet],
        loader: AudioLoader,
        segment_seconds: Optional[float] = None,
        train: bool = True,
        seed: int = 0,
    ) -> None:
        self.triplets = triplets
        self.loader = loader
        self.segment_seconds = segment_seconds
        self.train = train
        self.seed = seed

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        loader: AudioLoader,
        **kwargs: object,
    ) -> "GuidedSeparationDataset":
        with open(manifest_path, "r", encoding="utf-8") as f:
            triplets: List[Triplet] = json.load(f)
        return cls(triplets, loader, **kwargs)  # type: ignore[arg-type]

    def __len__(self) -> int:
        return len(self.triplets)

    @property
    def _segment_samples(self) -> Optional[int]:
        if self.segment_seconds is None:
            return None
        return int(self.segment_seconds * self.loader.sample_rate)

    def _crop(self, waveform: torch.Tensor, index: int) -> torch.Tensor:
        num_samples = self._segment_samples
        if num_samples is None:
            return waveform

        length = waveform.shape[-1]
        if length < num_samples:
            pad = num_samples - length
            return torch.nn.functional.pad(waveform, (0, pad))

        if length == num_samples:
            return waveform

        if self.train:
            generator = torch.Generator().manual_seed(self.seed + index)
            start = int(
                torch.randint(0, length - num_samples + 1, (1,), generator=generator)
            )
        else:
            start = 0
        return waveform[..., start : start + num_samples]

    def __getitem__(self, index: int) -> Sample:
        triplet = self.triplets[index]

        mixture = self.loader.load(triplet["mixture"])
        reference = self.loader.load(triplet["reference"])
        target = self.loader.load(triplet["target"])

        mixture = self._crop(mixture, index)
        target = self._crop(target, index)

        return Sample(mixture=mixture, reference=reference, target=target)
