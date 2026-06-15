"""STFTFeatureExtractor: waveform <-> espectrograma complejo (como tensor real)."""

from __future__ import annotations

import torch
from torch import nn
from typing import Optional

from src.models.components.feature_config import FeatureConfig


class STFTFeatureExtractor(nn.Module):
    """Calcula el STFT (one-sided) y su inversa con parametros compartidos."""

    def __init__(self, config: FeatureConfig) -> None:
        super().__init__()
        self.config = config
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.win_length = config.effective_win_length
        self.normalized = config.normalized

        window = torch.hann_window(self.win_length)
        self.register_buffer("window", window, persistent=False)

    def stft(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        complex_spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            normalized=self.normalized,
            return_complex=True,
        )
        return torch.view_as_real(complex_spec)

    def istft(
        self,
        spectrogram: torch.Tensor,
        length: Optional[int] = None,
    ) -> torch.Tensor:
        complex_spec = torch.view_as_complex(spectrogram.contiguous())
        return torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            normalized=self.normalized,
            length=length,
            return_complex=False,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.stft(waveform)
