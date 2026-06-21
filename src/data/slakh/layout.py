"""Adaptadores de layout para datasets tipo Slakh (BabySlakh / Slakh2100)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List


class Split(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass(frozen=True)
class StemEntry:
    key: str
    stem_id: str
    inst_class: str
    audio_path: Path
    midi_path: Path


@dataclass(frozen=True)
class TrackEntry:
    track_id: str
    track_dir: Path
    mix_path: Path
    stems: List[StemEntry]


class SlakhLayout(ABC):
    @abstractmethod
    def list_track_dirs(self, root: Path, split: Split) -> List[Path]:
        ...

    @abstractmethod
    def resolve_mix_path(self, track_dir: Path) -> Path:
        ...

    @abstractmethod
    def resolve_stem_paths(
        self, track_dir: Path, stem_id: str
    ) -> tuple[Path, Path]:
        ...


class BabySlakhLayout(SlakhLayout):
    """BabySlakh: tracks planos Track00001/ sin carpetas train/val/test."""

    def list_track_dirs(self, root: Path, split: Split) -> List[Path]:
        del split
        tracks = sorted(
            p for p in root.iterdir() if p.is_dir() and p.name.startswith("Track")
        )
        return tracks

    def resolve_mix_path(self, track_dir: Path) -> Path:
        wav = track_dir / "mix.wav"
        if wav.is_file():
            return wav
        flac = track_dir / "mix.flac"
        if flac.is_file():
            return flac
        raise FileNotFoundError(f"No mix file in {track_dir}")

    def resolve_stem_paths(
        self, track_dir: Path, stem_id: str
    ) -> tuple[Path, Path]:
        audio = track_dir / "stems" / f"{stem_id}.wav"
        if not audio.is_file():
            audio = track_dir / "stems" / f"{stem_id}.flac"
        midi = track_dir / "MIDI" / f"{stem_id}.mid"
        return audio, midi


class Slakh2100Layout(SlakhLayout):
    """Slakh2100 redux: train+omitted / validation / test."""

    _SPLIT_DIRS: Dict[Split, List[str]] = {
        Split.TRAIN: ["train", "omitted"],
        Split.VAL: ["validation"],
        Split.TEST: ["test"],
    }

    def list_track_dirs(self, root: Path, split: Split) -> List[Path]:
        track_dirs: List[Path] = []
        for sub in self._SPLIT_DIRS[split]:
            split_dir = root / sub
            if not split_dir.is_dir():
                continue
            track_dirs.extend(
                sorted(p for p in split_dir.iterdir() if p.is_dir())
            )
        return track_dirs

    def resolve_mix_path(self, track_dir: Path) -> Path:
        flac = track_dir / "mix.flac"
        if flac.is_file():
            return flac
        wav = track_dir / "mix.wav"
        if wav.is_file():
            return wav
        raise FileNotFoundError(f"No mix file in {track_dir}")

    def resolve_stem_paths(
        self, track_dir: Path, stem_id: str
    ) -> tuple[Path, Path]:
        audio = track_dir / "stems" / f"{stem_id}.flac"
        if not audio.is_file():
            audio = track_dir / "stems" / f"{stem_id}.wav"
        midi = track_dir / "MIDI" / f"{stem_id}.mid"
        return audio, midi


def get_layout(name: str) -> SlakhLayout:
    layouts = {
        "babyslakh": BabySlakhLayout(),
        "slakh2100": Slakh2100Layout(),
    }
    key = name.lower()
    if key not in layouts:
        raise ValueError(f"Unknown layout {name!r}. Choose from {list(layouts)}")
    return layouts[key]


def stem_is_usable(stem_meta: Dict[str, Any], audio_path: Path) -> bool:
    if stem_meta.get("is_drum", False):
        return False
    if stem_meta.get("inst_class") == "Drums":
        return False
    return audio_path.is_file() and audio_path.stat().st_size > 0


def split_tracks(
    track_dirs: Iterable[Path],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[Split, List[Path]]:
    import random

    tracks = sorted(track_dirs, key=lambda p: p.name)
    rng = random.Random(seed)
    rng.shuffle(tracks)

    n = len(tracks)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = max(0, n - n_train - n_val)
    if n_test == 0 and test_ratio > 0 and n > n_train + n_val:
        n_test = 1
        n_train = max(1, n_train - 1)

    return {
        Split.TRAIN: tracks[:n_train],
        Split.VAL: tracks[n_train : n_train + n_val],
        Split.TEST: tracks[n_train + n_val : n_train + n_val + n_test],
    }
