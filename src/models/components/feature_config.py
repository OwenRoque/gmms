"""Objetos de configuracion centralizados para el feature pipeline de GMSS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FeatureConfig:
    """Bundle inmutable de parametros de audio + STFT + mel-band.

    ``mono=False`` activa el modo estereo: el loader devuelve (C, T),
    el STFT produce (B, F*C, T, 2) y MelBandProjection / MaskEstimator
    operan con ``audio_channels = 2``.
    """

    # --- Audio / STFT ---
    sample_rate: int = 44100
    mono: bool = True           # False → estereo (audio_channels = 2)
    n_fft: int = 2048
    hop_length: int = 512
    win_length: Optional[int] = None
    normalized: bool = False

    # --- Band split ---
    num_bands: int = 60
    dim: int = 384

    # --- Guided conditioning (cross-attention) ---
    cross_attn_heads: int = 8
    cross_attn_dropout: float = 0.0

    # --- RoFormer backbone ---
    depth: int = 4
    time_transformer_depth: int = 2
    freq_transformer_depth: int = 2
    linear_transformer_depth: int = 1
    dim_head: int = 64
    heads: int = 8
    attn_dropout: float = 0.1
    ff_dropout: float = 0.1
    ff_mult: int = 4
    flash_attn: bool = True
    add_value_residual: bool = False   # True = valor residual aprendido entre capas

    # --- Mask estimator ---
    mask_estimator_depth: int = 2
    mlp_expansion_factor: int = 4

    # --- Loss ---
    multi_stft_loss_weight: float = 1.0

    @property
    def audio_channels(self) -> int:
        """1 para mono, 2 para estereo."""
        return 1 if self.mono else 2

    @property
    def stereo(self) -> bool:
        return not self.mono

    @property
    def effective_win_length(self) -> int:
        return self.win_length if self.win_length is not None else self.n_fft

    @property
    def num_freqs(self) -> int:
        return self.n_fft // 2 + 1
