"""GuidedSeparator: pipeline completo de separacion guiada por referencia.

Pipeline:
    mixture  ──► STFT ──► MelBandProjection ──► GuidedCrossAttention ──►
                                                  ▲
    reference ──► STFT ──► MelBandProjection ────┘
    (pesos compartidos con mixture)
    
    ──► RoFormerBackbone ──► MaskEstimator ──► apply_mel_band_masks ──► iSTFT

Notas de diseno:
  - STFT y MelBandProjection tienen pesos *compartidos* entre mezcla y
    referencia. Esto es correcto porque ambas senales pertenecen al mismo
    dominio espectral y queremos un espacio de representacion comun.
  - GuidedCrossAttention NO se modifica (condicionamiento drop-in).
  - MaskEstimator usa MLP+GLU (un unico stem, no multi-stem).
  - Soporta mono (audio_channels=1) y estereo (audio_channels=2).
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from torch import nn

from src.models.components.blocks.roformer_blocks import (
    MaskEstimator,
    RoFormerBackbone,
)
from src.models.components.conditioning.guided_cross_attention import (
    GuidedCrossAttention,
)
from src.models.components.feature_config import FeatureConfig
from src.models.components.features.mel_band import (
    MelBandProjection,
    apply_mel_band_masks,
)
from src.models.components.features.stft import STFTFeatureExtractor


class GuidedSeparator(nn.Module):
    """Separador guiado por referencia con backbone Mel-RoFormer.

    Args:
        config: FeatureConfig con todos los hiperparametros del modelo.
    """

    def __init__(self, config: FeatureConfig) -> None:
        super().__init__()
        self.config = config

        # --- Extraccion espectral (compartida para mezcla y referencia) ---
        self.stft = STFTFeatureExtractor(config)

        # --- Band split (pesos compartidos mezcla / referencia) ---
        self.band_split = MelBandProjection(config)

        # --- Condicionamiento guiado ---
        self.cross_attention = GuidedCrossAttention(config)

        # --- Backbone Mel-RoFormer ---
        self.backbone = RoFormerBackbone(config)

        # --- Estimador de mascara (un unico stem) ---
        self.mask_estimator = MaskEstimator(
            dim=config.dim,
            dim_inputs=self.band_split.freqs_per_bands_with_complex,
            depth=config.mask_estimator_depth,
            mlp_expansion_factor=config.mlp_expansion_factor,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        mixture: torch.Tensor,
        reference: torch.Tensor,
        output_length: Optional[int] = None,
        return_intermediate: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Separa el stem de interes de la mezcla guiado por la referencia.

        Args:
            mixture:    Forma (B, C, T) o (B, T).  C = audio_channels.
            reference:  Forma (B, C, S) o (B, S).  S puede diferir de T.
            output_length: Longitud objetivo del waveform de salida.
                           Si es None se usa mixture.shape[-1].
            return_intermediate: Si True, retorna ademas el espectrograma
                                 enmascarado y la mascara plana.

        Returns:
            separated:  (B, C, T) para estereo, (B, T) para mono.
            Si return_intermediate=True: (separated, masked_spec, mask_flat).
        """
        length = output_length if output_length is not None else mixture.shape[-1]

        # --- STFT ---
        mix_spec = self.stft(mixture)      # (B, F*C, T_frames, 2)
        ref_spec = self.stft(reference)    # (B, F*C, S_frames, 2)

        # --- Band split ---
        mix_feat = self.band_split(mix_spec)   # (B, T_frames, num_bands, D)
        ref_feat = self.band_split(ref_spec)   # (B, S_frames, num_bands, D)

        # --- Condicionamiento: referencia entra como contexto ---
        # GuidedCrossAttention: Q = mezcla, KV = referencia → (B, T, F, D)
        conditioned = self.cross_attention(mix_feat, ref_feat)

        # --- Backbone axial ---
        backbone_out = self.backbone(conditioned)   # (B, T, num_bands, D)

        # --- Estimacion de mascara ---
        mask_flat = self.mask_estimator(backbone_out)   # (B, T, sum(dim_inputs))

        # --- Aplicar mascara al espectrograma de mezcla ---
        masked_spec = apply_mel_band_masks(mix_spec, mask_flat, self.band_split)

        # --- iSTFT ---
        separated = self.stft.istft(masked_spec, length=length)

        if return_intermediate:
            return separated, masked_spec, mask_flat
        return separated

    def separate(
        self,
        mixture: torch.Tensor,
        reference: torch.Tensor,
        output_length: Optional[int] = None,
    ) -> torch.Tensor:
        """Alias publico de forward sin salidas intermedias."""
        return self.forward(  # type: ignore[return-value]
            mixture, reference, output_length=output_length, return_intermediate=False
        )
