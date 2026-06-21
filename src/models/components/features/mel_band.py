"""MelBandProjection y utilidades de band-split estilo Mel-RoFormer.

Cambios respecto a v1:
- MelBandProjection soporta estereo (audio_channels=2): freq_indices se
  duplican e intercalan, y cada banda recibe 4*n_bins en lugar de 2*n_bins.
- apply_mel_band_masks actualizado para manejar espectrogramas (B, F*C, T, 2).
- MelBandMaskEstimator (simple, sin MLP/GLU) conservado para MinimalSeparator.

La clase MaskEstimator con MLP+GLU esta en blocks/roformer_blocks.py.
"""

from __future__ import annotations

import librosa
import torch
from torch import nn
from typing import Tuple

from src.models.components.feature_config import FeatureConfig


# ---------------------------------------------------------------------------
# Norma
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(x, dim=-1) * self.scale * self.gamma


# ---------------------------------------------------------------------------
# Band Split
# ---------------------------------------------------------------------------

class MelBandProjection(nn.Module):
    """Proyecta cada banda mel del espectrograma complejo a dim features.

    Soporta mono (audio_channels=1) y estereo (audio_channels=2).  Para estereo
    los indices de frecuencia se intercalan: (f0_L, f0_R, f1_L, f1_R, ...).
    La forma de entrada esperada es (B, F*C, T_frames, 2) producida por
    STFTFeatureExtractor.

    Propiedades relevantes:
        total_bins      – bins mono cubiertos por el filtro mel (invariante a canales)
        total_indexed   – len(freq_indices) = total_bins * audio_channels
        freqs_per_bands_with_complex – tuple de dim_inputs para cada banda
    """

    def __init__(self, config: FeatureConfig) -> None:
        super().__init__()
        self.config = config
        self.num_bands = config.num_bands
        self.num_freqs = config.num_freqs
        self.audio_channels = config.audio_channels

        # --- Banco de filtros mel ---
        mel_fb_np = librosa.filters.mel(
            sr=config.sample_rate, n_fft=config.n_fft, n_mels=config.num_bands
        )
        mel_fb = torch.from_numpy(mel_fb_np)
        # Suavizado de extremos (recomendado en BS-RoFormer)
        mel_fb[0, 0] = mel_fb[0, 1] * 0.25
        mel_fb[-1, -1] = mel_fb[-1, -2] * 0.25

        freqs_per_band = mel_fb > 0
        assert freqs_per_band.any(dim=0).all(), (
            "every frequency bin must be covered by at least one band"
        )

        num_freqs_per_band = freqs_per_band.sum(dim=-1)  # (num_bands,)
        self.num_freqs_per_band: Tuple[int, ...] = tuple(
            num_freqs_per_band.tolist()
        )
        # Conteo mono de bins (siempre referenciado a F, no F*C)
        self._mono_total_bins: int = int(num_freqs_per_band.sum().item())

        # --- Indices de frecuencia ---
        band_ids, freq_ids = torch.where(freqs_per_band)
        order = torch.argsort(band_ids * self.num_freqs + freq_ids)
        freq_indices_mono = freq_ids[order]  # (total_bins,)

        if self.audio_channels == 2:
            # Intercalar canales L y R: f_L = 2*f, f_R = 2*f+1
            freq_indices_stereo = torch.stack(
                [freq_indices_mono * 2, freq_indices_mono * 2 + 1], dim=-1
            ).reshape(-1)                   # (total_bins*2,)
            self.register_buffer("freq_indices", freq_indices_stereo, persistent=False)
        else:
            self.register_buffer("freq_indices", freq_indices_mono, persistent=False)

        # num_bands_per_freq siempre en resolucion mono (F,)
        num_bands_per_freq = freqs_per_band.sum(dim=0)  # (F,)
        self.register_buffer("num_bands_per_freq", num_bands_per_freq, persistent=False)

        # dim_inputs por banda: 2*n_bins*audio_channels
        self.freqs_per_bands_with_complex: Tuple[int, ...] = tuple(
            2 * n * self.audio_channels for n in self.num_freqs_per_band
        )

        # Proyecciones lineales por banda
        self.to_features = nn.ModuleList(
            [
                nn.Sequential(RMSNorm(dim_in), nn.Linear(dim_in, config.dim))
                for dim_in in self.freqs_per_bands_with_complex
            ]
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spectrogram: (B, F*C, T_frames, 2)
        Returns:
            (B, T_frames, num_bands, dim)
        """
        batch = spectrogram.shape[0]
        # Recopilar bins indexados: (B, total_indexed, T, 2)
        gathered = spectrogram[:, self.freq_indices]
        # → (B, T, total_indexed, 2)
        gathered = gathered.permute(0, 2, 1, 3)
        # → (B, T, total_indexed*2)
        gathered = gathered.reshape(batch, gathered.shape[1], -1)

        chunks = gathered.split(list(self.freqs_per_bands_with_complex), dim=-1)
        band_features = [proj(chunk) for proj, chunk in zip(self.to_features, chunks)]
        return torch.stack(band_features, dim=-2)   # (B, T, num_bands, dim)

    # ------------------------------------------------------------------
    # Propiedades
    # ------------------------------------------------------------------

    @property
    def total_bins(self) -> int:
        """Bins mono cubiertos por el filtro (invariante a audio_channels)."""
        return self._mono_total_bins

    @property
    def total_indexed(self) -> int:
        """Numero total de indices en freq_indices (total_bins * audio_channels)."""
        return int(self.freq_indices.shape[0])


# ---------------------------------------------------------------------------
# Mask Estimator simple (conservado para MinimalSeparator)
# ---------------------------------------------------------------------------

class MelBandMaskEstimator(nn.Module):
    """Estimador de mascara lineal por banda (RMSNorm + Linear).

    Conservado por compatibilidad con MinimalSeparator / smoke tests.
    Para el backbone completo usa MaskEstimator en blocks/roformer_blocks.py.
    """

    def __init__(
        self,
        config: FeatureConfig,
        band_split: MelBandProjection,
        init_identity: bool = True,
    ) -> None:
        super().__init__()
        if band_split.num_bands != config.num_bands:
            raise ValueError("band_split.num_bands must match config.num_bands")

        self.config = config
        self.band_split = band_split

        self.to_masks = nn.ModuleList(
            [
                nn.Sequential(
                    RMSNorm(config.dim),
                    nn.Linear(config.dim, dim_in),
                )
                for dim_in in band_split.freqs_per_bands_with_complex
            ]
        )

        if init_identity:
            self._init_identity_masks()

    def _init_identity_masks(self) -> None:
        with torch.no_grad():
            for head in self.to_masks:
                linear = head[-1]
                assert isinstance(linear, nn.Linear)
                linear.weight.zero_()
                bias = linear.bias.view(-1, 2)
                bias[:, 0] = 1.0
                bias[:, 1] = 0.0

    def forward(self, band_features: torch.Tensor) -> torch.Tensor:
        bands = band_features.unbind(dim=-2)
        chunks = [head(feats) for head, feats in zip(self.to_masks, bands)]
        return torch.cat(chunks, dim=-1)


# ---------------------------------------------------------------------------
# Aplicar mascara al espectrograma (mono y estereo)
# ---------------------------------------------------------------------------

def apply_mel_band_masks(
    spectrogram: torch.Tensor,
    mask_flat: torch.Tensor,
    band_split: MelBandProjection,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Aplica la mascara compleja al espectrograma de la mezcla.

    Args:
        spectrogram: (B, F*C, T_frames, 2)  C = audio_channels
        mask_flat:   (B, T_frames, total_indexed*2)
        band_split:  instancia de MelBandProjection que provee freq_indices
                     y num_bands_per_freq.
    Returns:
        (B, F*C, T_frames, 2)  espectrograma enmascarado.
    """
    batch, num_freqs_full, num_frames, _ = spectrogram.shape
    total_indexed = band_split.total_indexed

    if mask_flat.shape[-1] != 2 * total_indexed:
        raise ValueError(
            f"mask_flat last dim {mask_flat.shape[-1]} "
            f"!= 2 * total_indexed ({2 * total_indexed})"
        )

    # Mascara compleja: (B, T, total_indexed)
    masks = mask_flat.reshape(batch, num_frames, total_indexed, 2)
    masks_c = torch.view_as_complex(masks.contiguous())

    # Espectrograma complejo: (B, F*C, T)
    spec_c = torch.view_as_complex(spectrogram.contiguous())

    masks_summed = torch.zeros(
        batch, num_freqs_full, num_frames,
        dtype=masks_c.dtype,
        device=masks_c.device,
    )

    # Acumular mascaras en las frecuencias correspondientes
    freq_idx = band_split.freq_indices.view(1, -1, 1).expand(batch, -1, num_frames)
    masks_summed.scatter_add_(1, freq_idx, masks_c.permute(0, 2, 1))

    # Denominador: cuantas bandas cubren cada bin de frecuencia
    nbpf = band_split.num_bands_per_freq  # (F,) mono
    if band_split.audio_channels == 2:
        # Expandir a (F*2,) intercalado: L y R comparten el mismo conteo
        nbpf = nbpf.repeat_interleave(2)

    denom = nbpf.clamp(min=eps).view(1, num_freqs_full, 1)
    masks_avg = masks_summed / denom.to(masks_summed.dtype)

    masked = spec_c * masks_avg
    return torch.view_as_real(masked)
