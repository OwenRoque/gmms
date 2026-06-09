"""MinimalSeparator: pipeline de separacion v1 (sin backbone RoFormer).

Flujo
-----
mixture waveform -> STFT -> MelBandProjection -> MelBandMaskEstimator
    -> apply_mel_band_masks -> iSTFT -> separated waveform

Es una primera version end-to-end: el mask estimator es un MLP por band sin
transformer intermedio. Con ``init_identity=True`` (default) la salida ~= mixture;
con entrenamiento futuro las mascaras pueden aprender a aislar fuentes.
"""

from __future__ import annotations

import torch
from torch import nn

from gmss.config import FeatureConfig
from gmss.features.mel_band import (
    MelBandMaskEstimator,
    MelBandProjection,
    apply_mel_band_masks,
)
from gmss.features.stft import STFTFeatureExtractor


class MinimalSeparator(nn.Module):
    """Separador minimo mono: estima mascara compleja y reconstruye waveform.

    Parameters
    ----------
    config:
        :class:`FeatureConfig` compartido por todas las etapas.
    init_identity_mask:
        Si ``True``, el mask estimator arranca en identidad (1+0j).
    """

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
        output_length: int | None = None,
        return_spec: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Separar la mixture y devolver waveform reconstruido.

        Parameters
        ----------
        mixture:
            ``[B, T]`` o ``[T]`` mono.
        output_length:
            Longitud de salida del iSTFT; default = longitud de la mixture.
        return_spec:
            Si ``True``, devuelve ``(waveform, masked_spec, mask_flat)``.

        Returns
        -------
        torch.Tensor or tuple
            Waveform ``[B, T]`` o tupla con espectrograma enmascarado y mascara.
        """
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

    def separate(self, mixture: torch.Tensor, output_length: int | None = None) -> torch.Tensor:
        """Alias explicito de :meth:`forward` (sin extras)."""
        return self.forward(mixture, output_length=output_length, return_spec=False)  # type: ignore[return-value]
