"""Generacion on-the-fly de muestras {target, reference, residual}."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import torch

from src.data.processing.midi_utils import extract_midi_segment
from src.data.processing.waveform_utils import (
    load_waveform_segment,
    peak_normalize,
    signal_entropy,
)
from src.data.slakh.humtrans import HumTransStore
from src.data.slakh.index import SlakhTrackIndex, TrackEntry
from src.data.slakh.moisesdb import MoisesResidualSampler
from src.data.slakh.synthesis import ReferenceSynthesizer


@dataclass
class RawSample:
    target: torch.Tensor
    reference: torch.Tensor
    residual: torch.Tensor
    target_instrument: str = ""
    reference_instrument: str = ""
    track_id: str = ""


class SampleBuilder:
    """Orquesta la logica de GuideSep sin extraer features espectrales."""

    def __init__(
        self,
        sample_rate: int,
        segment_seconds: float,
        moises: MoisesResidualSampler,
        synthesizer: ReferenceSynthesizer,
        humtrans: Optional[HumTransStore] = None,
        humming_prob: float = 0.4,
        org_mix_prob: float = 0.3,
        augment: bool = True,
        mode: str = "train",
        tar_ins: Optional[str] = None,
        min_entropy: float = 1.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.segment_seconds = segment_seconds
        self.moises = moises
        self.synthesizer = synthesizer
        self.humtrans = humtrans
        self.humming_prob = humming_prob
        self.org_mix_prob = org_mix_prob
        self.augment = augment
        self.mode = mode
        self.tar_ins = tar_ins
        self.min_entropy = min_entropy

    def _load_target(
        self, track: TrackEntry, stem_key: str, start_time: float
    ) -> torch.Tensor:
        stem = next(s for s in track.stems if s.key == stem_key)
        waveform, _ = load_waveform_segment(
            filepath=stem.audio_path,
            sample_rate=self.sample_rate,
            segment_seconds=self.segment_seconds,
            start=start_time,
            random_start=False,
        )
        return waveform

    def _load_mix_residual(
        self, track: TrackEntry, target: torch.Tensor, start_time: float
    ) -> torch.Tensor:
        mixture, _ = load_waveform_segment(
            filepath=track.mix_path,
            sample_rate=self.sample_rate,
            segment_seconds=self.segment_seconds,
            start=start_time,
            random_start=False,
        )
        return mixture - target

    def build_humming_from_midi(self, midi_path: str) -> RawSample:
        target, reference, residual, tar_ins, ref_name = self._build_humming(midi_path)
        return RawSample(
            target=target,
            reference=reference,
            residual=residual,
            target_instrument=tar_ins,
            reference_instrument=ref_name,
            track_id="humtrans",
        )

    def _build_humming(
        self, midi_path: Optional[str] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, str]:
        assert self.humtrans is not None
        if midi_path is None:
            midi_path = self.humtrans.sample_midi_path()
        midi, pitch_range, start_time = extract_midi_segment(
            midi_file=midi_path,
            segment_seconds=self.segment_seconds,
            augment=False,
        )
        wav_path = self.humtrans.wav_path_for_midi(midi_path)
        reference, _ = load_waveform_segment(
            filepath=wav_path,
            sample_rate=self.sample_rate,
            segment_seconds=self.segment_seconds,
            start=start_time,
            random_start=False,
        )
        target, ref_name = self.synthesizer.synthesize(
            midi=midi,
            pitch_range=pitch_range,
            segment_seconds=self.segment_seconds,
        )
        num_sources = random.randint(3, 5)
        residual = self.moises.sample_sum(num_sources)
        return target, reference, residual, "humming", ref_name

    def build_from_track(
        self,
        track: TrackEntry,
        stem_key: Optional[str] = None,
        start_time: Optional[float] = None,
        midi=None,
        pitch_range: Optional[list[int]] = None,
    ) -> RawSample:
        use_humming = (
            self.humtrans is not None
            and self.humtrans.available
            and self.mode == "train"
            and random.random() < self.humming_prob
        )
        use_org_mix = random.random() < self.org_mix_prob

        if use_humming:
            target, reference, residual, tar_ins, ref_name = self._build_humming(None)
        else:
            if stem_key is None:
                if self.mode in ("test", "val") and self.tar_ins:
                    filtered = SlakhTrackIndex.filter_stems_by_class(
                        track, self.tar_ins
                    )
                    if not filtered:
                        raise ValueError(
                            f"No stems matching {self.tar_ins!r} in {track.track_id}"
                        )
                    stem = random.choice(filtered)
                else:
                    stem = random.choice(track.stems)
                stem_key = stem.key

            stem = next(s for s in track.stems if s.key == stem_key)
            tar_ins = stem.key

            if midi is None:
                midi, pitch_range, start_time = extract_midi_segment(
                    midi_file=str(stem.midi_path),
                    segment_seconds=self.segment_seconds,
                    augment=self.augment and self.mode == "train",
                )

            reference, ref_name = self.synthesizer.synthesize(
                midi=midi,
                pitch_range=pitch_range,
                segment_seconds=self.segment_seconds,
            )
            target = self._load_target(track, stem_key, start_time)

            synth_mix = (not use_org_mix and self.mode == "train") or use_humming
            if synth_mix and self.moises.available:
                num_sources = random.randint(3, 5)
                residual = self.moises.sample_sum(num_sources)
            else:
                residual = self._load_mix_residual(track, target, start_time)

        reference = peak_normalize(reference)
        target = peak_normalize(target)
        residual = peak_normalize(residual)

        if signal_entropy(reference) < self.min_entropy:
            reference = target.clone()
        if signal_entropy(target) < self.min_entropy:
            target = reference.clone()

        return RawSample(
            target=target,
            reference=reference,
            residual=residual,
            target_instrument=tar_ins if not use_humming else "humming",
            reference_instrument=ref_name,
            track_id=track.track_id,
        )

    def build(self, track: TrackEntry) -> RawSample:
        return self.build_from_track(track)
