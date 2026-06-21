"""Utilidades de carga y normalizacion de waveforms (segmentos alineados)."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn.functional as F
import torchaudio
from julius import ResampleFrac
from scipy.stats import entropy


def signal_entropy(wav: torch.Tensor) -> float:
    values, counts = torch.unique(wav, return_counts=True)
    if values.numel() <= 1:
        return 0.0
    probabilities = counts.float() / counts.sum()
    return float(entropy(probabilities.cpu().numpy()))


def peak_normalize(waveform: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    peak = waveform.abs().max()
    if peak < eps:
        return waveform
    return waveform / peak


def load_waveform_segment(
    filepath: Union[str, Path],
    sample_rate: int,
    segment_seconds: float,
    start: Optional[float] = None,
    random_start: bool = True,
) -> tuple[torch.Tensor, float]:
    """Carga un segmento mono 1D alineado temporalmente.

    Returns:
        (waveform, start_time_seconds)
    """
    filepath = Path(filepath)
    metadata = torchaudio.info(str(filepath))
    src_sr = metadata.sample_rate
    src_frames = metadata.num_frames
    segment_frames = int(segment_seconds * src_sr)

    if segment_frames <= 0:
        raise ValueError("segment_seconds must be positive")

    if start is None:
        if random_start and src_frames > segment_frames:
            start_frame = random.randint(0, src_frames - segment_frames)
        else:
            start_frame = 0
        start_time = start_frame / src_sr
    else:
        start_frame = int(start * src_sr)
        start_time = start
        if start_frame >= src_frames:
            zeros = torch.zeros(int(segment_seconds * sample_rate))
            return zeros, start_time

    waveform, _ = torchaudio.load(
        str(filepath),
        frame_offset=start_frame,
        num_frames=segment_frames,
    )
    waveform = waveform.mean(dim=0)

    if waveform.shape[-1] < segment_frames:
        waveform = F.pad(waveform, (0, segment_frames - waveform.shape[-1]))

    if src_sr != sample_rate:
        waveform = ResampleFrac(src_sr, sample_rate)(waveform.unsqueeze(0)).squeeze(0)

    target_samples = int(segment_seconds * sample_rate)
    if waveform.shape[-1] < target_samples:
        waveform = F.pad(waveform, (0, target_samples - waveform.shape[-1]))
    elif waveform.shape[-1] > target_samples:
        waveform = waveform[:target_samples]

    return waveform.contiguous(), start_time
