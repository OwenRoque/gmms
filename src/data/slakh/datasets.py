"""Datasets on-the-fly para separacion guiada."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import Dataset

from src.data.processing.midi_utils import split_midi_by_interval
from src.data.slakh.index import SlakhTrackIndex, TrackEntry
from src.data.slakh.sample_builder import RawSample, SampleBuilder


@dataclass
class TestSegment:
    track_idx: int
    stem_key: str
    start_time: float
    pitch_range: List[int]
    midi: object


class SlakhGuidedTrainDataset(Dataset):
    def __init__(
        self,
        index: SlakhTrackIndex,
        builder: SampleBuilder,
    ) -> None:
        self.index = index
        self.builder = builder

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> RawSample:
        return self.builder.build(self.index.get(idx))


class SlakhGuidedValDataset(SlakhGuidedTrainDataset):
    pass


class SlakhGuidedTestDataset(Dataset):
    """Expande cada track/instrumento en segmentos MIDI fijos."""

    def __init__(
        self,
        index: SlakhTrackIndex,
        builder: SampleBuilder,
        segment_seconds: float,
        tar_ins: str,
        augment: bool = False,
    ) -> None:
        self.index = index
        self.builder = builder
        self.segment_seconds = segment_seconds
        self.tar_ins = tar_ins
        self.augment = augment
        self.segments: List[TestSegment] = self._build_segments()

    def _build_segments(self) -> List[TestSegment]:
        segments: List[TestSegment] = []
        for track_idx, track in enumerate(self.index.tracks):
            filtered = SlakhTrackIndex.filter_stems_by_class(track, self.tar_ins)
            for stem in filtered:
                midi_list, pitch_ranges, start_times = split_midi_by_interval(
                    midi_file=str(stem.midi_path),
                    segment_seconds=self.segment_seconds,
                    augment=self.augment,
                )
                for midi, pitch_range, start_time in zip(
                    midi_list, pitch_ranges, start_times
                ):
                    segments.append(
                        TestSegment(
                            track_idx=track_idx,
                            stem_key=stem.key,
                            start_time=start_time,
                            pitch_range=pitch_range,
                            midi=midi,
                        )
                    )
        return segments

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> RawSample:
        seg = self.segments[idx]
        track = self.index.get(seg.track_idx)
        return self.builder.build_from_track(
            track=track,
            stem_key=seg.stem_key,
            start_time=seg.start_time,
            midi=seg.midi,
            pitch_range=seg.pitch_range,
        )


class HumTransTestDataset(Dataset):
    """Evaluacion humming: condicion HumTrans, target sintetizado, residual Moises."""

    def __init__(
        self,
        humtrans_midi_files: List[str],
        builder: SampleBuilder,
    ) -> None:
        self.midi_files = humtrans_midi_files
        self.builder = builder

    def __len__(self) -> int:
        return len(self.midi_files)

    def __getitem__(self, idx: int) -> RawSample:
        return self.builder.build_humming_from_midi(self.midi_files[idx])
