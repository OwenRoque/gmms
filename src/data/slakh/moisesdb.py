"""Sampler de stems residuales desde MoisesDB."""

from __future__ import annotations

import glob
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from src.data.processing.waveform_utils import load_waveform_segment, signal_entropy


class MoisesResidualSampler:
    def __init__(
        self,
        root: Path,
        sample_rate: int,
        segment_seconds: float,
        min_entropy: float = 2.0,
        max_attempts: int = 20,
    ) -> None:
        self.sample_rate = sample_rate
        self.segment_seconds = segment_seconds
        self.min_entropy = min_entropy
        self.max_attempts = max_attempts
        self._stems_by_key: Dict[str, List[str]] = self._index(root)
        self._keys = np.array(list(self._stems_by_key.keys()))

    @staticmethod
    def _index(root: Path) -> Dict[str, List[str]]:
        stems_by_key: Dict[str, List[str]] = {}
        if not root.is_dir():
            return stems_by_key

        for track in root.iterdir():
            if not track.is_dir():
                continue
            for stem_dir in track.iterdir():
                if not stem_dir.is_dir():
                    continue
                key = stem_dir.name
                sources = glob.glob(str(stem_dir / "*.wav"))
                if not sources:
                    continue
                stems_by_key.setdefault(key, []).extend(sources)
        return stems_by_key

    @property
    def available(self) -> bool:
        return len(self._keys) > 0

    def sample(self) -> torch.Tensor:
        if not self.available:
            raise RuntimeError("MoisesDB index is empty")

        for _ in range(self.max_attempts):
            key = random.choice(self._keys)
            filepath = random.choice(self._stems_by_key[key])
            waveform, _ = load_waveform_segment(
                filepath=filepath,
                sample_rate=self.sample_rate,
                segment_seconds=self.segment_seconds,
                random_start=True,
            )
            if signal_entropy(waveform) > self.min_entropy:
                return waveform

        waveform, _ = load_waveform_segment(
            filepath=random.choice(self._stems_by_key[random.choice(self._keys)]),
            sample_rate=self.sample_rate,
            segment_seconds=self.segment_seconds,
            random_start=True,
        )
        return waveform

    def sample_sum(self, num_sources: int) -> torch.Tensor:
        return torch.stack([self.sample() for _ in range(num_sources)]).sum(dim=0)
