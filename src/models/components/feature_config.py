"""Objetos de configuracion centralizados para el feature pipeline de GMSS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FeatureConfig:
    """Bundle inmutable de parametros de audio + STFT + mel-band."""

    sample_rate: int = 44100
    mono: bool = True
    n_fft: int = 2048
    hop_length: int = 512
    win_length: Optional[int] = None
    normalized: bool = False
    num_bands: int = 60
    dim: int = 384

    @property
    def effective_win_length(self) -> int:
        return self.win_length if self.win_length is not None else self.n_fft

    @property
    def num_freqs(self) -> int:
        return self.n_fft // 2 + 1
