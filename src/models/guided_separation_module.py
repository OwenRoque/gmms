"""GuidedSeparationLitModule: LightningModule de entrenamiento/validacion/test.

Implementa el ciclo completo de entrenamiento para separacion guiada por
referencia con el backbone Mel-RoFormer.

Loss:
  total = L1(separated, target)
        + multi_stft_loss_weight * sum_w L1(STFT_w(separated), STFT_w(target))

Logging: train/loss, val/loss, test/loss (compatible con W&B / CSV logger).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from lightning import LightningModule

from src.models.components.feature_config import FeatureConfig
from src.models.components.guided_separator import GuidedSeparator
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

# Tamanos de ventana para la loss multi-resolucion STFT (igual al original)
_MULTI_STFT_WINDOW_SIZES: Tuple[int, ...] = (4096, 2048, 1024, 512, 256)
_MULTI_STFT_HOP_SIZE: int = 147


class GuidedSeparationLitModule(LightningModule):
    """LightningModule para separacion de fuentes guiada.

    Args:
        feature:            FeatureConfig con todos los hiperparametros.
        lr:                 Tasa de aprendizaje para Adam.
        weight_decay:       Regularizacion L2.
        multi_stft_loss_weight: Peso de la loss multi-resolucion STFT.
                            0.0 desactiva esa componente.
    """

    def __init__(
        self,
        feature: FeatureConfig,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        multi_stft_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.separator = GuidedSeparator(feature)
        self._multi_stft_loss_weight = multi_stft_loss_weight

    # ------------------------------------------------------------------
    # Forward publico (usado en inferencia)
    # ------------------------------------------------------------------

    def forward(
        self,
        mixture: torch.Tensor,
        reference: torch.Tensor,
        output_length: Optional[int] = None,
    ) -> torch.Tensor:
        return self.separator(mixture, reference, output_length=output_length)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    @staticmethod
    def _align_channels(
        separated: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reconcilia la dimension de canal (convencion BS-RoFormer).

        El separador mono devuelve (B, T) y el estereo (B, C, T). El target
        del collator es siempre (B, T). Igualamos el numero de dimensiones
        agregando un eje de canal cuando hace falta, y validamos que el
        numero de canales coincida para evitar broadcast silencioso.
        """
        if separated.ndim == 3 and target.ndim == 2:
            target = target.unsqueeze(1)            # (B, 1, T)
        elif target.ndim == 3 and separated.ndim == 2:
            separated = separated.unsqueeze(1)

        if separated.shape[:-1] != target.shape[:-1]:
            raise ValueError(
                f"Desajuste de forma separated={tuple(separated.shape)} vs "
                f"target={tuple(target.shape)}. Verifica que data.feature.mono "
                f"y model.feature.mono coincidan."
            )
        return separated, target

    def _compute_loss(
        self,
        separated: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calcula L1 + multi-resolucion STFT loss.

        Returns:
            (total_loss, l1_loss, mr_stft_loss)
        """
        separated, target = self._align_channels(separated, target)

        # Asegura longitudes iguales (istft puede truncar un sample)
        min_len = min(separated.shape[-1], target.shape[-1])
        separated = separated[..., :min_len]
        target = target[..., :min_len]

        l1 = F.l1_loss(separated, target)

        mr_stft = torch.tensor(0.0, device=separated.device)

        if self._multi_stft_loss_weight > 0.0:
            # Aplanar todas las dimensiones de lote/canal: (..., T) -> (N, T)
            # Tras _align_channels separated y target comparten forma, por lo
            # que N es identico en ambos (igual que el rearrange del original).
            s_flat = separated.reshape(-1, min_len)
            t_flat = target.reshape(-1, min_len)

            for win_size in _MULTI_STFT_WINDOW_SIZES:
                n_fft = max(win_size, self.hparams.feature.n_fft)
                hop = max(_MULTI_STFT_HOP_SIZE, win_size // 4)
                window = torch.hann_window(win_size, device=separated.device)

                s_stft = torch.stft(
                    s_flat, n_fft=n_fft, win_length=win_size, hop_length=hop,
                    window=window, return_complex=True, normalized=False,
                )
                t_stft = torch.stft(
                    t_flat, n_fft=n_fft, win_length=win_size, hop_length=hop,
                    window=window, return_complex=True, normalized=False,
                )
                mr_stft = mr_stft + F.l1_loss(
                    torch.view_as_real(s_stft),
                    torch.view_as_real(t_stft),
                )

        total = l1 + self._multi_stft_loss_weight * mr_stft
        return total, l1, mr_stft

    # ------------------------------------------------------------------
    # Pasos de entrenamiento
    # ------------------------------------------------------------------

    def _shared_step(
        self,
        batch: Dict[str, torch.Tensor],
        stage: str,
    ) -> torch.Tensor:
        mixture = batch["mixture"]
        reference = batch["reference"]
        target = batch["target"]

        separated = self.separator(
            mixture, reference, output_length=mixture.shape[-1]
        )
        total, l1, mr = self._compute_loss(separated, target)

        on_step = stage == "train"
        bs = mixture.shape[0]
        self.log(f"{stage}/loss",      total, on_step=on_step, on_epoch=True,
                 prog_bar=True,  batch_size=bs)
        self.log(f"{stage}/l1_loss",   l1,    on_step=False,   on_epoch=True,
                 batch_size=bs)
        self.log(f"{stage}/mr_stft",   mr,    on_step=False,   on_epoch=True,
                 batch_size=bs)
        return total

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        mixture = batch["mixture"]
        reference = batch["reference"]
        target = batch["target"]

        separated, masked_spec, mask_flat = self.separator(
            mixture, reference,
            output_length=mixture.shape[-1],
            return_intermediate=True,
        )
        total, l1, mr = self._compute_loss(separated, target)

        bs = mixture.shape[0]
        self.log("test/loss",    total, batch_size=bs)
        self.log("test/l1_loss", l1,    batch_size=bs)
        self.log("test/mr_stft", mr,    batch_size=bs)

        log.info(
            f"\n[GuidedSeparator test]"
            f"\n  mixture     {tuple(mixture.shape)}"
            f"\n  reference   {tuple(reference.shape)}"
            f"\n  separated   {tuple(separated.shape)}"
            f"\n  mask_flat   {tuple(mask_flat.shape)}"
            f"\n  loss={total.item():.4f}  l1={l1.item():.4f}"
            f"  mr_stft={mr.item():.4f}"
        )
        return total

    # ------------------------------------------------------------------
    # Optimizador
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.Adam(
            self.separator.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return {"optimizer": optimizer}
