"""Indexacion de tracks Slakh desde metadata.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml

from src.data.slakh.layout import (
    SlakhLayout,
    Split,
    StemEntry,
    TrackEntry,
    get_layout,
    split_tracks,
    stem_is_usable,
)


class SlakhTrackIndex:
    """Catalogo de tracks y stems validos para generacion on-the-fly."""

    def __init__(
        self,
        root: Path,
        layout: SlakhLayout | str,
        split: Split,
        split_config: Optional[dict] = None,
    ) -> None:
        self.root = Path(root)
        self.layout = get_layout(layout) if isinstance(layout, str) else layout
        self.split = split
        self.split_config = split_config or {}
        self.tracks: List[TrackEntry] = self._build_index()

    def _resolve_track_dirs(self) -> List[Path]:
        strategy = self.split_config.get("strategy", "ratio")
        if strategy == "explicit":
            key = {
                Split.TRAIN: "train_tracks",
                Split.VAL: "val_tracks",
                Split.TEST: "test_tracks",
            }[self.split]
            names = self.split_config.get(key, [])
            return [self.root / name for name in names]

        all_tracks = self.layout.list_track_dirs(self.root, Split.TRAIN)
        if strategy == "ratio":
            ratios = self.split_config
            partitioned = split_tracks(
                all_tracks,
                train_ratio=float(ratios.get("train_ratio", 0.8)),
                val_ratio=float(ratios.get("val_ratio", 0.1)),
                test_ratio=float(ratios.get("test_ratio", 0.1)),
                seed=int(ratios.get("seed", 42)),
            )
            return partitioned[self.split]

        return self.layout.list_track_dirs(self.root, self.split)

    def _build_index(self) -> List[TrackEntry]:
        tracks: List[TrackEntry] = []
        for track_dir in self._resolve_track_dirs():
            if not track_dir.is_dir():
                continue
            metadata_path = track_dir / "metadata.yaml"
            if not metadata_path.is_file():
                continue

            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = yaml.safe_load(f)

            stems: List[StemEntry] = []
            count = 0
            for stem_id, info in metadata.get("stems", {}).items():
                audio_path, midi_path = self.layout.resolve_stem_paths(
                    track_dir, stem_id
                )
                if not stem_is_usable(info, audio_path):
                    continue
                if not midi_path.is_file():
                    continue
                inst_class = info.get("inst_class", stem_id)
                key = f"{inst_class}_{count}"
                stems.append(
                    StemEntry(
                        key=key,
                        stem_id=stem_id,
                        inst_class=inst_class,
                        audio_path=audio_path,
                        midi_path=midi_path,
                    )
                )
                count += 1

            if not stems:
                continue

            try:
                mix_path = self.layout.resolve_mix_path(track_dir)
            except FileNotFoundError:
                continue

            tracks.append(
                TrackEntry(
                    track_id=track_dir.name,
                    track_dir=track_dir,
                    mix_path=mix_path,
                    stems=stems,
                )
            )
        return tracks

    def __len__(self) -> int:
        return len(self.tracks)

    def get(self, idx: int) -> TrackEntry:
        return self.tracks[idx]

    @staticmethod
    def filter_stems_by_class(
        track: TrackEntry, tar_ins: str
    ) -> List[StemEntry]:
        return [s for s in track.stems if tar_ins.lower() in s.key.lower()]
