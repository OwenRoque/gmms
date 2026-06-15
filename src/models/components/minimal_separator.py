"""MinimalSeparator: pipeline de separacion v1 (sin backbone RoFormer)."""

from __future__ import annotations

import torch
from torch import nn
from typing import Optional, Tuple, Union

from src.models.components.feature_config import FeatureConfig
from src.models.components.features.mel_band import (
    MelBandMaskEstimator,
    MelBandProjection,
    apply_mel_band_masks,
)
from src.models.components.features.stft import STFTFeatureExtractor


class MinimalSeparator(nn.Module):
    def __init__(
        self,
        config: FeatureConfig,
        init_identity_mask: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.stft = STFTFeatureExtractor(config)
        self.band_split = MelBandProjection(config)
        self.mask_estimator = MelBandMaskEstimator(
            config,
            self.band_split,
            init_identity=init_identity_mask,
        )

    def forward(
        self,
        mixture: torch.Tensor,
        output_length: Optional[int] = None,
        return_spec: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if mixture.ndim == 1:
            mixture = mixture.unsqueeze(0)

        length = output_length if output_length is not None else mixture.shape[-1]

        spec = self.stft(mixture)
        band_features = self.band_split(spec)
        mask_flat = self.mask_estimator(band_features)
        masked_spec = apply_mel_band_masks(spec, mask_flat, self.band_split)
        separated = self.stft.istft(masked_spec, length=length)

        if return_spec:
            return separated, masked_spec, mask_flat
        return separated

    def separate(self, mixture: torch.Tensor, output_length: Optional[int] = None) -> torch.Tensor:
        return self.forward(mixture, output_length=output_length, return_spec=False)  # type: ignore[return-value]
