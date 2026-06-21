"""Sintesis MIDI con FluidSynth y seleccion de timbres Aegean."""

from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from pretty_midi import Instrument, PrettyMIDI

from src.data.processing.waveform_utils import signal_entropy


@dataclass
class InstrumentSpec:
    element: ET.Element
    name: str
    pitch_range: List[int]


def _extract_program_and_controller(
    instrument_element: ET.Element,
) -> tuple[int, int, str]:
    channels = instrument_element.findall("Channel")
    selected = random.choice(channels)
    ctrl_value = int(selected.find("controller").attrib["value"])
    program_value = int(selected.find("program").attrib["value"])
    channel_name = selected.get("name", "default")
    return ctrl_value, program_value, channel_name


def load_instrument_specs(xml_path: Path) -> List[InstrumentSpec]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    instruments: List[InstrumentSpec] = []
    for group in root.findall("InstrumentGroup"):
        for instrument in group.findall("Instrument"):
            long_name = instrument.find("longName").text
            pitch_elem = instrument.find("pPitchRange")
            if pitch_elem is not None and pitch_elem.text:
                low, high = pitch_elem.text.split("-")
                pitch_range = [int(low), int(high)]
            else:
                pitch_range = [0, 127]
            instruments.append(
                InstrumentSpec(
                    element=instrument,
                    name=long_name,
                    pitch_range=pitch_range,
                )
            )
    return instruments


class FluidSynthBackend:
    """Wrapper lazy de FluidSynth (seguro con num_workers=0 por defecto)."""

    def __init__(
        self,
        soundfont_path: Path,
        sample_rate: int,
    ) -> None:
        self.soundfont_path = Path(soundfont_path)
        self.sample_rate = sample_rate
        self._synth = None
        self._sfid: Optional[int] = None

    def _ensure_loaded(self) -> None:
        if self._synth is not None:
            return
        try:
            from fluidsynth import Synth
        except ImportError as exc:
            raise ImportError(
                "pyfluidsynth is required for MIDI synthesis. "
                "Install with: pip install pyfluidsynth"
            ) from exc

        self._synth = Synth(samplerate=float(self.sample_rate))
        self._sfid = self._synth.sfload(str(self.soundfont_path))

    def synthesize(
        self,
        midi: Union[str, PrettyMIDI],
        segment_seconds: float,
        program: int,
        controller: int,
        channel: int = 0,
    ) -> torch.Tensor:
        self._ensure_loaded()
        assert self._synth is not None and self._sfid is not None

        midi_obj = PrettyMIDI(midi) if isinstance(midi, str) else midi
        target_len = int(segment_seconds * self.sample_rate)

        instruments: List[Instrument] = midi_obj.instruments
        if not instruments or all(len(i.notes) == 0 for i in instruments):
            return torch.zeros(target_len, dtype=torch.float32)

        self._synth.program_select(channel, self._sfid, program, controller)
        waveforms: List[np.ndarray] = []

        for instrument in instruments:
            self._synth.program_change(0, program)
            event_list = []
            for note in instrument.notes:
                event_list.append([note.start, "note on", note.pitch, note.velocity])
                event_list.append([note.end, "note off", note.pitch])
            for bend in instrument.pitch_bends:
                event_list.append([bend.time, "pitch bend", bend.pitch])
            for cc in instrument.control_changes:
                event_list.append(
                    [cc.time, "control change", cc.number, cc.value]
                )

            event_list.sort(key=lambda x: (x[0], x[1] != "note off"))
            if not event_list:
                continue

            current_time = event_list[0][0]
            next_times = [e[0] for e in event_list[1:]]
            for event, end in zip(event_list[:-1], next_times):
                event[0] = end - event[0]
            event_list[-1][0] = 1.0

            total_time = current_time + sum(e[0] for e in event_list)
            synthesized = np.zeros(int(np.ceil(self.sample_rate * total_time)))

            for event in event_list:
                if event[1] == "note on":
                    self._synth.noteon(channel, event[2], event[3])
                elif event[1] == "note off":
                    self._synth.noteoff(channel, event[2])
                elif event[1] == "pitch bend":
                    self._synth.pitch_bend(channel, event[2])
                elif event[1] == "control change":
                    self._synth.cc(channel, event[2], event[3])

                current_sample = int(self.sample_rate * current_time)
                end_sample = int(self.sample_rate * (current_time + event[0]))
                samples = self._synth.get_samples(end_sample - current_sample)[::2]
                synthesized[current_sample:end_sample] += samples
                current_time += event[0]

            waveforms.append(synthesized)

        if not waveforms:
            return torch.zeros(target_len, dtype=torch.float32)

        out = np.zeros(max(w.shape[0] for w in waveforms))
        for w in waveforms:
            out[: w.shape[0]] += w

        if out.shape[-1] < target_len:
            out = np.pad(out, (0, target_len - out.shape[-1]))
        else:
            out = out[:target_len]

        out = out / (np.max(np.abs(out)) + 1e-10)
        return torch.tensor(out, dtype=torch.float32)


class ReferenceSynthesizer:
    """Selecciona timbre ASO compatible y sintetiza la referencia."""

    def __init__(
        self,
        backend: FluidSynthBackend,
        instruments: List[InstrumentSpec],
        min_entropy: float = 1.5,
    ) -> None:
        self.backend = backend
        self.instruments = instruments
        self.min_entropy = min_entropy

    def synthesize(
        self,
        midi: PrettyMIDI,
        pitch_range: List[int],
        segment_seconds: float,
    ) -> tuple[torch.Tensor, str]:
        candidates = self.instruments.copy()
        cond_name = "default"

        while candidates:
            selected = random.choice(candidates)
            low, high = selected.pitch_range
            if low <= pitch_range[0] and high >= pitch_range[1]:
                ctrl, program, channel_name = _extract_program_and_controller(
                    selected.element
                )
                cond_name = f"{selected.name}_{channel_name}"
                audio = self.backend.synthesize(
                    midi=midi,
                    segment_seconds=segment_seconds,
                    program=program,
                    controller=ctrl,
                )
                if signal_entropy(audio) < self.min_entropy and len(candidates) > 1:
                    candidates.remove(selected)
                    continue
                return audio, cond_name
            candidates.remove(selected)

        ctrl, program, channel_name = _extract_program_and_controller(
            self.instruments[0].element
        )
        cond_name = f"{self.instruments[0].name}_{channel_name}"
        audio = self.backend.synthesize(
            midi=midi,
            segment_seconds=segment_seconds,
            program=program,
            controller=ctrl,
        )
        return audio, cond_name
