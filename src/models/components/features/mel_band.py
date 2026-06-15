"""MelBandProjection: band split estilo Mel-RoFormer (NO es un mel spectrogram)."""

from __future__ import annotations

import librosa
import torch
from torch import nn
from typing import Tuple

from src.models.components.feature_config import FeatureConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(x, dim=-1) * self.scale * self.gamma


class MelBandProjection(nn.Module):
    def __init__(self, config: FeatureConfig) -> None:
        super().__init__()
        self.config = config
        self.num_bands = config.num_bands
        self.num_freqs = config.num_freqs

        mel_fb_np = librosa.filters.mel(
            sr=config.sample_rate, n_fft=config.n_fft, n_mels=config.num_bands
        )
        mel_fb = torch.from_numpy(mel_fb_np)
        mel_fb[0, 0] = mel_fb[0, 1] * 0.25
        mel_fb[-1, -1] = mel_fb[-1, -2] * 0.25

        freqs_per_band = mel_fb > 0
        assert freqs_per_band.any(dim=0).all(), (
            "every frequency bin must be covered by at least one band"
        )

        num_freqs_per_band = freqs_per_band.sum(dim=-1)
        self.num_freqs_per_band: Tuple[int, ...] = tuple(
            num_freqs_per_band.tolist()
        )

        band_ids, freq_ids = torch.where(freqs_per_band)
        order = torch.argsort(band_ids * self.num_freqs + freq_ids)
        freq_indices = freq_ids[order]
        self.register_buffer("freq_indices", freq_indices, persistent=False)

        num_bands_per_freq = freqs_per_band.sum(dim=0)
        self.register_buffer(
            "num_bands_per_freq", num_bands_per_freq, persistent=False
        )

        self.to_features = nn.ModuleList(
            [
                nn.Sequential(RMSNorm(2 * n_bins), nn.Linear(2 * n_bins, config.dim))
                for n_bins in self.num_freqs_per_band
            ]
        )

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        batch = spectrogram.shape[0]
        gathered = spectrogram[:, self.freq_indices]
        gathered = gathered.permute(0, 2, 1, 3).reshape(batch, -1, self._total_flat)

        split_sizes = [2 * n for n in self.num_freqs_per_band]
        chunks = gathered.split(split_sizes, dim=-1)
        band_features = [proj(chunk) for proj, chunk in zip(self.to_features, chunks)]
        return torch.stack(band_features, dim=-2)

    @property
    def total_bins(self) -> int:
        return int(self.freq_indices.shape[0])

    @property
    def _total_flat(self) -> int:
        return 2 * self.total_bins


class MelBandMaskEstimator(nn.Module):
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
                    nn.Linear(config.dim, 2 * n_bins),
                )
                for n_bins in band_split.num_freqs_per_band
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


def apply_mel_band_masks(
    spectrogram: torch.Tensor,
    mask_flat: torch.Tensor,
    band_split: MelBandProjection,
    eps: float = 1e-8,
) -> torch.Tensor:
    batch, num_freqs, num_frames, _ = spectrogram.shape
    total_bins = band_split.total_bins

    if mask_flat.shape[-1] != 2 * total_bins:
        raise ValueError(
            f"mask_flat last dim {mask_flat.shape[-1]} != 2 * total_bins ({2 * total_bins})"
        )

    masks = mask_flat.reshape(batch, num_frames, total_bins, 2)
    masks_c = torch.view_as_complex(masks.contiguous())
    spec_c = torch.view_as_complex(spectrogram.contiguous())

    masks_summed = torch.zeros(
        batch, num_freqs, num_frames,
        dtype=masks_c.dtype,
        device=masks_c.device,
    )

    freq_idx = band_split.freq_indices.view(1, -1, 1).expand(batch, -1, num_frames)
    masks_summed.scatter_add_(1, freq_idx, masks_c.permute(0, 2, 1))

    denom = band_split.num_bands_per_freq.clamp(min=eps).view(1, num_freqs, 1)
    masks_avg = masks_summed / denom.to(masks_summed.dtype)

    masked = spec_c * masks_avg
    return torch.view_as_real(masked)
