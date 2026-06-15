from typing import Any, Dict, List, Optional

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.data.audio_loader import AudioLoader
from src.data.dataset import GuidedSeparationDataset, Sample, Triplet
from src.models.components.feature_config import FeatureConfig


class GuideSepDataModule(LightningDataModule):
    """LightningDataModule para el smoke test de separacion guiada."""

    def __init__(
        self,
        triplets: List[Triplet],
        feature: FeatureConfig,
        segment_seconds: Optional[float] = 3.0,
        train: bool = True,
        seed: int = 0,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.feature = feature
        self.loader = AudioLoader(
            sample_rate=feature.sample_rate,
            mono=feature.mono,
        )
        self.dataset: Optional[GuidedSeparationDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is not None:
            return

        self.dataset = GuidedSeparationDataset(
            triplets=self.hparams.triplets,
            loader=self.loader,
            segment_seconds=self.hparams.segment_seconds,
            train=self.hparams.train,
            seed=self.hparams.seed,
        )

    def _collate(self, batch: List[Sample]) -> Dict[str, torch.Tensor]:
        return {
            key: torch.stack([sample[key] for sample in batch])
            for key in ("mixture", "reference", "target")
        }

    def test_dataloader(self) -> DataLoader[Any]:
        if self.dataset is None:
            raise RuntimeError("Call setup() before requesting test_dataloader().")

        return DataLoader(
            dataset=self.dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self._collate,
        )
