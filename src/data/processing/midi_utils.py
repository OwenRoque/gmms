"""Utilidades MIDI portadas de GuideSep (segmentacion y augmentacion)."""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import pretty_midi
from pretty_midi import Instrument, PrettyMIDI


def _apply_pitch_shift(note: pretty_midi.Note, semitone_shift: int) -> pretty_midi.Note:
    note.pitch = max(0, min(127, note.pitch + semitone_shift))
    return note


def _apply_time_shift(
    note: pretty_midi.Note,
    start_time_shift: float,
    end_time_shift: float,
    current_time_shift: float,
) -> tuple[pretty_midi.Note, float]:
    shift_time = 0.0
    new_start_time = start_time_shift + note.start + current_time_shift
    if new_start_time < 0:
        new_start_time = 0.0
        start_time_shift = (-note.start) - current_time_shift
    new_end_time = note.end + end_time_shift + current_time_shift
    if new_end_time > new_start_time:
        shift_time = start_time_shift
        note.end = new_end_time
        note.start = new_start_time
    return note, shift_time


def _apply_velocity_modulation(note: pretty_midi.Note, velocity_jitter: int) -> pretty_midi.Note:
    note.velocity = max(0, min(127, note.velocity + velocity_jitter))
    return note


def _apply_octave_shift(notes: list[pretty_midi.Note], octave_shift: int) -> list[pretty_midi.Note]:
    if not notes:
        return notes
    pitches = [note.pitch for note in notes]
    if max(pitches) + octave_shift * 12 > 127 or min(pitches) + octave_shift * 12 < 0:
        return notes
    return [_apply_pitch_shift(note, octave_shift * 12) for note in notes]


def augment_midi(
    midi_data: PrettyMIDI,
    num_note: int,
    max_octave_shift: int = 1,
    max_pitch_bend: int = 20,
    max_time_shift: float = 0.03,
    velocity_jitter_range: int = 5,
    octave_shift_prob: float = 0.5,
    pitch_bend_prob: float = 0.5,
    time_shift_prob: float = 0.4,
    velocity_jitter_prob: float = 0.3,
    note_drop_prob: float = 0.5,
    use_note_drop_prob: float = 0.7,
) -> tuple[PrettyMIDI, int, int]:
    low_pitch = 200
    high_pitch = -1

    for instrument in midi_data.instruments:
        notes: list[pretty_midi.Note] = []
        current_time_shift = 0.0
        drop_note = random.random() < use_note_drop_prob

        for note in instrument.notes:
            random_floats = np.random.rand(4)

            if num_note > 4 and drop_note and random_floats[0] < note_drop_prob:
                continue

            if random_floats[1] < pitch_bend_prob:
                pitch_bend = np.random.uniform(-max_pitch_bend, max_pitch_bend)
                bend_amount = pitch_bend * 8192 / 100
                instrument.pitch_bends.append(
                    pretty_midi.PitchBend(int(bend_amount), note.start)
                )

            if random_floats[2] < time_shift_prob:
                start_shift = np.random.uniform(-max_time_shift, max_time_shift)
                end_shift = np.random.uniform(-max_time_shift, max_time_shift)
                note, time_shift = _apply_time_shift(
                    note, start_shift, end_shift, current_time_shift
                )
                current_time_shift += time_shift

            if random_floats[3] < velocity_jitter_prob:
                jitter = np.random.randint(
                    -velocity_jitter_range, velocity_jitter_range + 1
                )
                note = _apply_velocity_modulation(note, jitter)

            notes.append(note)
            high_pitch = max(high_pitch, note.pitch)
            low_pitch = min(low_pitch, note.pitch)

        if random.random() < octave_shift_prob and notes:
            octave_shift = np.random.randint(-max_octave_shift, max_octave_shift + 1)
            notes = _apply_octave_shift(notes, octave_shift)

        instrument.notes = notes

    return midi_data, low_pitch, high_pitch


def extract_midi_segment(
    midi_file: str,
    segment_seconds: float,
    melody_only: bool = True,
    start_time: Optional[float] = None,
    augment: bool = False,
) -> tuple[PrettyMIDI, list[int], float]:
    pretty_midi_obj = pretty_midi.PrettyMIDI(midi_file)
    new_midi = pretty_midi.PrettyMIDI()

    if start_time is None:
        all_notes = [
            note
            for instrument in pretty_midi_obj.instruments
            for note in instrument.notes
        ]
        if all_notes:
            start_time = random.choice(all_notes).start
        else:
            start_time = random.uniform(10, 30)

    end_time = start_time + segment_seconds
    low_pitch = 200
    high_pitch = -1

    for instrument in pretty_midi_obj.instruments:
        new_instrument = pretty_midi.Instrument(
            program=instrument.program,
            is_drum=instrument.is_drum,
            name=instrument.name,
        )
        for note in instrument.notes:
            note_start = max(note.start, start_time)
            note_end = min(note.end, end_time)
            if note_end > note_start:
                new_note = pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note_start - start_time,
                    end=note_end - start_time,
                )
                new_instrument.notes.append(new_note)
                high_pitch = max(high_pitch, note.pitch)
                low_pitch = min(low_pitch, note.pitch)
        new_midi.instruments.append(new_instrument)

    if melody_only:
        low_pitch = 200
        high_pitch = -1
        melody_midi = pretty_midi.PrettyMIDI()
        for instrument in new_midi.instruments:
            melody_notes: list[pretty_midi.Note] = []
            new_instrument = pretty_midi.Instrument(
                program=instrument.program,
                is_drum=instrument.is_drum,
                name=instrument.name,
            )
            instrument.notes.sort(key=lambda note: (note.start, -note.pitch))
            current_time = -np.inf
            for note in instrument.notes:
                if note.start > current_time:
                    melody_notes.append(note)
                    high_pitch = max(high_pitch, note.pitch)
                    low_pitch = min(low_pitch, note.pitch)
                    current_time = note.start
            new_instrument.notes = melody_notes
            melody_midi.instruments.append(new_instrument)
        new_midi = melody_midi

    num_note = sum(len(inst.notes) for inst in new_midi.instruments)
    if augment and num_note > 0:
        drop_note_prob = 0.0 if melody_only else 0.7
        new_midi, low_pitch, high_pitch = augment_midi(
            new_midi, num_note, use_note_drop_prob=drop_note_prob
        )

    return new_midi, [low_pitch, high_pitch], start_time


def split_midi_by_interval(
    midi_file: str,
    segment_seconds: float,
    melody_only: bool = True,
    augment: bool = False,
) -> tuple[list[PrettyMIDI], list[list[int]], list[float]]:
    midi_obj = pretty_midi.PrettyMIDI(midi_file)
    all_notes = [note for instrument in midi_obj.instruments for note in instrument.notes]
    if not all_notes:
        return [], [], []

    all_notes.sort(key=lambda note: note.start)
    split_midi_objs: list[PrettyMIDI] = []
    pitch_range_list: list[list[int]] = []
    start_time_list: list[float] = []

    current_time = all_notes[0].start
    current_notes: list[pretty_midi.Note] = []
    low_pitch = 200
    high_pitch = -1

    for note in all_notes:
        if note.start >= current_time + segment_seconds:
            new_midi = pretty_midi.PrettyMIDI()
            new_instrument = pretty_midi.Instrument(
                program=midi_obj.instruments[0].program,
                is_drum=midi_obj.instruments[0].is_drum,
                name=midi_obj.instruments[0].name,
            )
            anchor = current_notes[0].start
            for n in current_notes:
                new_instrument.notes.append(
                    pretty_midi.Note(
                        velocity=n.velocity,
                        pitch=n.pitch,
                        start=n.start - anchor,
                        end=n.end - anchor,
                    )
                )
            new_midi.instruments.append(new_instrument)
            start_time_list.append(anchor)
            pitch_range_list.append([low_pitch, high_pitch])

            if augment:
                num_note = len(new_instrument.notes)
                drop_note_prob = 0.0 if melody_only else 0.7
                new_midi, low_pitch, high_pitch = augment_midi(
                    new_midi, num_note, use_note_drop_prob=drop_note_prob
                )

            split_midi_objs.append(new_midi)
            current_time = note.start
            current_notes = []
            low_pitch = 200
            high_pitch = -1

        current_notes.append(note)
        high_pitch = max(high_pitch, note.pitch)
        low_pitch = min(low_pitch, note.pitch)

    if current_notes:
        new_midi = pretty_midi.PrettyMIDI()
        new_instrument = pretty_midi.Instrument(program=0)
        anchor = current_notes[0].start
        for n in current_notes:
            new_instrument.notes.append(
                pretty_midi.Note(
                    velocity=n.velocity,
                    pitch=n.pitch,
                    start=n.start - anchor,
                    end=n.end - anchor,
                )
            )
        new_midi.instruments.append(new_instrument)
        split_midi_objs.append(new_midi)
        start_time_list.append(anchor)
        pitch_range_list.append([low_pitch, high_pitch])

    return split_midi_objs, pitch_range_list, start_time_list
