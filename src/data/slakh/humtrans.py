"""HumTrans: extraccion lazy y sampler de pares MIDI/WAV."""

from __future__ import annotations

import glob
import json
import random
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class HumTransStore:
    """Gestiona extraccion y acceso a archivos HumTrans."""

    def __init__(
        self,
        root: Path,
        auto_extract: bool = True,
        extract_wav: bool = True,
    ) -> None:
        self.root = Path(root)
        self.auto_extract = auto_extract
        self.extract_wav = extract_wav
        self.midi_dir = self.root / "midi_data"
        self.wav_dir = self.root / "wav_data_sync_with_midi"
        self._midi_by_key: Dict[str, List[str]] = {}
        self._keys: np.ndarray = np.array([])

    def _resolve_dirs(self) -> None:
        """Localiza midi_data/ y wav_data_sync_with_midi/ en layouts comunes.

        Tras descomprimir, los archivos pueden quedar en:
          root/midi_data/                      o  root/all_midi/midi_data/
          root/wav_data_sync_with_midi/        o  root/all_wav/wav_data_sync_with_midi/
        """
        midi_candidates = [
            self.root / "midi_data",
            self.root / "all_midi" / "midi_data",
        ]
        wav_candidates = [
            self.root / "wav_data_sync_with_midi",
            self.root / "all_wav" / "wav_data_sync_with_midi",
        ]
        for cand in midi_candidates:
            if cand.is_dir() and any(cand.glob("*.mid")):
                self.midi_dir = cand
                break
        for cand in wav_candidates:
            if cand.is_dir() and any(cand.glob("*.wav")):
                self.wav_dir = cand
                break

    def prepare(self) -> None:
        # Resolver primero evita re-extraer archivos ya descomprimidos.
        self._resolve_dirs()
        if self.auto_extract:
            self._maybe_extract()
            self._resolve_dirs()
        self._build_index()

    def _maybe_extract(self) -> None:
        midi_zip = self.root / "all_midi.zip"
        wav_zip = self.root / "all_wav.zip"

        if not self.midi_dir.is_dir() and midi_zip.is_file():
            log.info(f"Extracting HumTrans MIDI archive to {self.midi_dir}")
            with zipfile.ZipFile(midi_zip, "r") as zf:
                zf.extractall(self.root)

        if (
            self.extract_wav
            and wav_zip.is_file()
            and (not self.wav_dir.is_dir() or not any(self.wav_dir.glob("*.wav")))
        ):
            log.info(
                f"Extracting HumTrans WAV archive to {self.wav_dir} "
                "(this may take several minutes)..."
            )
            with zipfile.ZipFile(wav_zip, "r") as zf:
                zf.extractall(self.root)

    def _build_index(self) -> None:
        midi_files = glob.glob(str(self.midi_dir / "*.mid"))
        self._midi_by_key = {}
        for filepath in midi_files:
            name = Path(filepath).stem
            parts = name.split("_")
            key = parts[1] if len(parts) > 1 else name
            self._midi_by_key.setdefault(key, []).append(filepath)

        self._keys = np.array(list(self._midi_by_key.keys()))

    @property
    def available(self) -> bool:
        return len(self._keys) > 0 and self.wav_dir.is_dir()

    def wav_path_for_midi(self, midi_path: str) -> Path:
        name = Path(midi_path).stem + ".wav"
        return self.wav_dir / name

    def sample_midi_path(self) -> str:
        if not self.available:
            raise RuntimeError("HumTrans is not available")
        key = random.choice(self._keys)
        return random.choice(self._midi_by_key[key])

    @staticmethod
    def load_split_keys(split_file: Path, split: str) -> Optional[List[str]]:
        if not split_file.is_file():
            return None
        with open(split_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get(split.upper())
