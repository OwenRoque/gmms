from typing import Any, Dict

import torch
from lightning import LightningModule

from src.models.components.feature_config import FeatureConfig
from src.models.components.minimal_separator import MinimalSeparator
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class SeparationSmokeLitModule(LightningModule):
    """Smoke test del pipeline mixture -> mask -> iSTFT via PyTorch Lightning."""

    def __init__(
        self,
        feature: FeatureConfig,
        init_identity_mask: bool = True,
        case_name: str = "",
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.feature = feature
        self.case_name = case_name
        self.separator = MinimalSeparator(feature, init_identity_mask=init_identity_mask)

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        return self.separator(mixture)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        mixture = batch["mixture"]
        separated, masked_spec, mask_flat = self.separator(
            mixture,
            output_length=mixture.shape[-1],
            return_spec=True,
        )
        rmse = torch.mean((separated - mixture) ** 2).sqrt()

        label = self.case_name or (
            "init_identity=True" if self.hparams.init_identity_mask else "init_identity=False"
        )
        log.info(f"\n--- {label} ---")
        log.info(f"    mixture     {tuple(mixture.shape)}")
        log.info(f"    separated   {tuple(separated.shape)}")
        log.info(f"    masked_spec {tuple(masked_spec.shape)}")
        log.info(f"    mask_flat   {tuple(mask_flat.shape)}")
        log.info(f"    RMSE vs mixture = {rmse.item():.2e}")

        metric_key = (
            "test/rmse_identity" if self.hparams.init_identity_mask else "test/rmse_random"
        )
        self.log(metric_key, rmse, prog_bar=True, batch_size=mixture.shape[0])
        return rmse

    def configure_optimizers(self) -> Dict[str, Any]:
        raise NotImplementedError("SeparationSmokeLitModule is inference-only.")
