"""AudioLoader: carga robusta de waveforms basada en torchaudio."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import soundfile as sf
import torch
import torchaudio


class AudioLoader:
    def __init__(
        self,
        sample_rate: int = 44100,
        mono: bool = True,
        normalize: bool = True,
    ) -> None:
        self.sample_rate = sample_rate
        self.mono = mono
        self.normalize = normalize
        self._resamplers: Dict[int, torchaudio.transforms.Resample] = {}

    def _get_resampler(self, orig_sr: int) -> torchaudio.transforms.Resample:
        if orig_sr not in self._resamplers:
            self._resamplers[orig_sr] = torchaudio.transforms.Resample(
                orig_freq=orig_sr, new_freq=self.sample_rate
            )
        return self._resamplers[orig_sr]

    @staticmethod
    def _to_mono(waveform: torch.Tensor) -> torch.Tensor:
        if waveform.shape[0] == 1:
            return waveform
        return waveform.mean(dim=0, keepdim=True)

    def _resample(self, waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
        if orig_sr == self.sample_rate:
            return waveform
        return self._get_resampler(orig_sr)(waveform)

    @staticmethod
    def _peak_normalize(waveform: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        peak = waveform.abs().max()
        if peak < eps:
            return waveform
        return waveform / peak

    def load(self, path: str | Path) -> torch.Tensor:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")

        data, orig_sr = sf.read(str(path), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T).contiguous()

        if self.mono:
            waveform = self._to_mono(waveform)

        waveform = self._resample(waveform, orig_sr)

        if self.normalize:
            waveform = self._peak_normalize(waveform)

        if self.mono:
            waveform = waveform.squeeze(0)

        return waveform.contiguous()
