"""Lightning DataModule on-the-fly estilo GuideSep para GMSS."""

from __future__ import annotations

import glob
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.data.slakh.collator import CollatorConfig, GuidedSeparationCollator
from src.data.slakh.datasets import (
    HumTransTestDataset,
    SlakhGuidedTestDataset,
    SlakhGuidedTrainDataset,
    SlakhGuidedValDataset,
)
from src.data.slakh.humtrans import HumTransStore
from src.data.slakh.index import SlakhTrackIndex
from src.data.slakh.layout import Split
from src.data.slakh.moisesdb import MoisesResidualSampler
from src.data.slakh.sample_builder import SampleBuilder
from src.data.slakh.synthesis import (
    FluidSynthBackend,
    ReferenceSynthesizer,
    load_instrument_specs,
)
from src.models.components.feature_config import FeatureConfig
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


@dataclass
class SplitConfig:
    strategy: str = "ratio"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    train_tracks: list[str] = field(default_factory=list)
    val_tracks: list[str] = field(default_factory=list)
    test_tracks: list[str] = field(default_factory=list)


def _coerce_dataclass(cls, value: Optional[Union[Mapping[str, Any], Any]]) -> Any:
    if value is None:
        return cls()
    if isinstance(value, cls):
        return value
    if isinstance(value, Mapping):
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in dict(value).items() if k in valid})
    return cls()


class SlakhGuidedDataModule(LightningDataModule):
    """DataModule on-the-fly compatible con GuidedSeparationLitModule."""

    def __init__(
        self,
        slakh_root: str,
        layout: str = "babyslakh",
        moisesdb_root: str = "",
        humtrans_root: Optional[str] = None,
        soundfont_path: str = "",
        instruments_xml: str = "",
        feature: FeatureConfig = FeatureConfig(),
        segment_seconds: float = 4.0,
        humming_prob: float = 0.4,
        org_mix_prob: float = 0.3,
        augment: bool = True,
        tar_ins: str = "Piano",
        auto_extract_humtrans: bool = True,
        extract_humtrans_wav: bool = True,
        batch_size: int = 2,
        num_workers: int = 0,
        pin_memory: bool = False,
        collator: Optional[Union[CollatorConfig, Mapping[str, Any]]] = None,
        split: Optional[Union[SplitConfig, Mapping[str, Any]]] = None,
        test_humming: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.feature = feature
        self.segment_seconds = segment_seconds
        self.collator_cfg = _coerce_dataclass(CollatorConfig, collator)
        split_cfg = _coerce_dataclass(SplitConfig, split)
        self.split_dict = {
            "strategy": split_cfg.strategy,
            "train_ratio": split_cfg.train_ratio,
            "val_ratio": split_cfg.val_ratio,
            "test_ratio": split_cfg.test_ratio,
            "seed": split_cfg.seed,
            "train_tracks": split_cfg.train_tracks,
            "val_tracks": split_cfg.val_tracks,
            "test_tracks": split_cfg.test_tracks,
        }

        self.slakh_root = Path(slakh_root)
        self.moisesdb_root = Path(moisesdb_root) if moisesdb_root else None
        self.humtrans_root = Path(humtrans_root) if humtrans_root else None
        self.soundfont_path = Path(soundfont_path)
        self.instruments_xml = Path(instruments_xml)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self._humtrans: Optional[HumTransStore] = None
        self._moises: Optional[MoisesResidualSampler] = None
        self._synth_backend: Optional[FluidSynthBackend] = None
        self._ref_synth: Optional[ReferenceSynthesizer] = None

    @property
    def remix(self) -> bool:
        return self.hparams.org_mix_prob > 0 and self.collator_cfg.remix

    def prepare_data(self) -> None:
        if self.humtrans_root is not None:
            store = HumTransStore(
                root=self.humtrans_root,
                auto_extract=self.hparams.auto_extract_humtrans,
                extract_wav=self.hparams.extract_humtrans_wav,
            )
            store.prepare()

    def _shared_resources(self) -> tuple[
        MoisesResidualSampler,
        ReferenceSynthesizer,
        Optional[HumTransStore],
    ]:
        if self._moises is None:
            assert self.moisesdb_root is not None
            self._moises = MoisesResidualSampler(
                root=self.moisesdb_root,
                sample_rate=self.feature.sample_rate,
                segment_seconds=self.segment_seconds,
            )
        if self._synth_backend is None:
            self._synth_backend = FluidSynthBackend(
                soundfont_path=self.soundfont_path,
                sample_rate=self.feature.sample_rate,
            )
        if self._ref_synth is None:
            instruments = load_instrument_specs(self.instruments_xml)
            self._ref_synth = ReferenceSynthesizer(
                backend=self._synth_backend,
                instruments=instruments,
            )
        if self._humtrans is None and self.humtrans_root is not None:
            self._humtrans = HumTransStore(
                root=self.humtrans_root,
                auto_extract=False,
            )
            self._humtrans.prepare()

        assert self._moises is not None and self._ref_synth is not None
        return self._moises, self._ref_synth, self._humtrans

    def _make_builder(
        self,
        mode: str,
        humming_prob: float,
        org_mix_prob: float,
        augment: bool,
    ) -> SampleBuilder:
        moises, ref_synth, humtrans = self._shared_resources()
        return SampleBuilder(
            sample_rate=self.feature.sample_rate,
            segment_seconds=self.segment_seconds,
            moises=moises,
            synthesizer=ref_synth,
            humtrans=humtrans,
            humming_prob=humming_prob,
            org_mix_prob=org_mix_prob,
            augment=augment,
            mode=mode,
            tar_ins=self.hparams.tar_ins,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        del stage
        if self.data_train is not None:
            return

        train_index = SlakhTrackIndex(
            root=self.slakh_root,
            layout=self.hparams.layout,
            split=Split.TRAIN,
            split_config=self.split_dict,
        )
        val_index = SlakhTrackIndex(
            root=self.slakh_root,
            layout=self.hparams.layout,
            split=Split.VAL,
            split_config=self.split_dict,
        )
        test_index = SlakhTrackIndex(
            root=self.slakh_root,
            layout=self.hparams.layout,
            split=Split.TEST,
            split_config=self.split_dict,
        )

        log.info(
            f"Slakh index sizes — train: {len(train_index)}, "
            f"val: {len(val_index)}, test: {len(test_index)}"
        )

        train_builder = self._make_builder(
            mode="train",
            humming_prob=self.hparams.humming_prob,
            org_mix_prob=self.hparams.org_mix_prob,
            augment=self.hparams.augment,
        )
        val_builder = self._make_builder(
            mode="val",
            humming_prob=0.0,
            org_mix_prob=1.0,
            augment=False,
        )
        test_builder = self._make_builder(
            mode="test",
            humming_prob=0.0,
            org_mix_prob=1.0,
            augment=False,
        )

        self.data_train = SlakhGuidedTrainDataset(train_index, train_builder)
        self.data_val = SlakhGuidedValDataset(val_index, val_builder)

        if self.hparams.test_humming and self.humtrans_root is not None:
            _, _, humtrans = self._shared_resources()
            assert humtrans is not None
            midi_files = sorted(glob.glob(str(humtrans.midi_dir / "*.mid")))
            humming_builder = self._make_builder(
                mode="test",
                humming_prob=1.0,
                org_mix_prob=0.0,
                augment=False,
            )
            self.data_test = HumTransTestDataset(midi_files, humming_builder)
        else:
            self.data_test = SlakhGuidedTestDataset(
                index=test_index,
                builder=test_builder,
                segment_seconds=self.segment_seconds,
                tar_ins=self.hparams.tar_ins,
                augment=False,
            )

    def _dataloader(
        self,
        dataset: Dataset,
        shuffle: bool,
        mode: str,
        remix: bool,
    ) -> DataLoader:
        collator = GuidedSeparationCollator(
            mode=mode,
            config=CollatorConfig(
                snr_mean=self.collator_cfg.snr_mean,
                snr_std=self.collator_cfg.snr_std,
                snr_min=self.collator_cfg.snr_min,
                snr_max=self.collator_cfg.snr_max,
                remix=remix,
            ),
        )
        return DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collator,
            drop_last=mode == "train",
        )

    def train_dataloader(self) -> DataLoader:
        assert self.data_train is not None
        return self._dataloader(
            self.data_train, shuffle=True, mode="train", remix=self.remix
        )

    def val_dataloader(self) -> DataLoader:
        assert self.data_val is not None
        return self._dataloader(
            self.data_val, shuffle=False, mode="val", remix=False
        )

    def test_dataloader(self) -> DataLoader:
        assert self.data_test is not None
        return self._dataloader(
            self.data_test, shuffle=False, mode="test", remix=False
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        del stage

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        del state_dict
