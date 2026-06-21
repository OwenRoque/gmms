"""STFTFeatureExtractor: waveform <-> espectrograma complejo (mono y estereo).

Convencion de forma para estereo (audio_channels=2):
    - STFT  input:  (B, C, T)   → output: (B, F*C, T_frames, 2)   [canales intercalados en F]
    - iSTFT input:  (B, F*C, T_frames, 2) → output: (B, C, T)
    - Interleaving: (f0_L, f0_R, f1_L, f1_R, ...)  equivalente a original MelBandRoformer.

Para mono (audio_channels=1) el comportamiento es identico al anterior:
    - STFT  input:  (B, T) o (T,) → output: (B, F, T_frames, 2)
    - iSTFT input:  (B, F, T_frames, 2)  → output: (B, T)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from typing import Optional

from src.models.components.feature_config import FeatureConfig


class STFTFeatureExtractor(nn.Module):
    """Calcula el STFT (one-sided) y su inversa con parametros compartidos.

    Acepta mono ``(B, T)`` o estereo ``(B, C, T)``.  Si el numero de canales
    del tensor de entrada no coincide con ``config.audio_channels``, los
    canales se duplican automaticamente (ej. WAV mono con config estereo).
    """

    def __init__(self, config: FeatureConfig) -> None:
        super().__init__()
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.win_length = config.effective_win_length
        self.normalized = config.normalized
        self.audio_channels = config.audio_channels

        window = torch.hann_window(self.win_length)
        self.register_buffer("window", window, persistent=False)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _to_BCT(self, waveform: torch.Tensor) -> torch.Tensor:
        """Normaliza a forma (B, C, T) y expande canales si hace falta."""
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)   # (1, 1, T)
        elif waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)                # (B, 1, T)
        # waveform es (B, C_in, T)
        C_in = waveform.shape[1]
        if C_in < self.audio_channels:
            waveform = waveform.expand(
                waveform.shape[0], self.audio_channels, waveform.shape[2]
            ).contiguous()
        return waveform

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def stft(self, waveform: torch.Tensor) -> torch.Tensor:
        """Calcula STFT con canales intercalados en la dimension de frecuencia.

        Returns:
            (B, F*C, T_frames, 2)  donde C = audio_channels.
        """
        x = self._to_BCT(waveform)          # (B, C, T)
        B, C, T_len = x.shape

        wav_flat = x.reshape(B * C, T_len)
        spec_complex = torch.stft(
            wav_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,             # type: ignore[arg-type]
            normalized=self.normalized,
            return_complex=True,
        )                                   # (B*C, F, T_frames)

        spec = torch.view_as_real(spec_complex)          # (B*C, F, T_frames, 2)
        spec = spec.reshape(B, C, spec.shape[1], spec.shape[2], 2)
        # Intercalar canales: (B, C, F, T, 2) → (B, F*C, T, 2)
        # para C=1 esto es un no-op efectivo
        spec = rearrange(spec, "b c f t ri -> b (f c) t ri")
        return spec

    def istft(
        self,
        spectrogram: torch.Tensor,
        length: Optional[int] = None,
    ) -> torch.Tensor:
        """Invierte el STFT desde forma intercalada.

        Args:
            spectrogram: (B, F*C, T_frames, 2)
        Returns:
            mono:   (B, T)
            estereo: (B, C, T)
        """
        C = self.audio_channels

        if C == 1:
            spec_c = torch.view_as_complex(spectrogram.contiguous())  # (B, F, T)
            audio = torch.istft(
                spec_c,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,         # type: ignore[arg-type]
                normalized=self.normalized,
                length=length,
                return_complex=False,
            )                               # (B, T)
            return audio

        # Estereo: des-intercalar → (B, C, F, T, 2)
        B = spectrogram.shape[0]
        spec_deint = rearrange(spectrogram, "b (f c) t ri -> b c f t ri", c=C)
        spec_c = torch.view_as_complex(spec_deint.contiguous())        # (B, C, F, T)
        spec_flat = spec_c.reshape(B * C, spec_c.shape[2], spec_c.shape[3])
        audio_flat = torch.istft(
            spec_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,             # type: ignore[arg-type]
            normalized=self.normalized,
            length=length,
            return_complex=False,
        )                                   # (B*C, T)
        return audio_flat.reshape(B, C, -1) # (B, C, T)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.stft(waveform)
