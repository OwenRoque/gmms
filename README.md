# GMSS — Guided Music Source Separation

A guided source-separation system inspired by **Mel-Band RoFormer**, adapted to
take a **reference/condition** signal (e.g. a hummed melody) as in
**User-Guided Generative Source Separation**.

This repository currently implements **only the data pipeline and feature
extraction** stages. 

<!-- No model backbone, training loop, diffusion, multi-resSTFT, or FluidSynth yet. -->

```
waveform (mixture) ─┐
waveform (reference)─┤→ AudioLoader → STFTFeatureExtractor → MelBandProjection → [B, T, bands, dim]
waveform (target) ──┘
```

## Project structure

```
gmss/
├── configs/
│   └── default.yaml              # mirrors FeatureConfig
├── scripts/
│   └── test_pipeline.py          # loads one sample, prints tensor shapes
├── src/gmss/
│   ├── config.py                 # FeatureConfig (shared STFT/mel params)
│   ├── data/
│   │   ├── audio_loader.py       # AudioLoader: load → (mono) → resample → normalize
│   │   └── dataset.py            # GuidedSeparationDataset (mixture/reference/target)
│   └── features/
│       ├── stft.py               # STFTFeatureExtractor (forward + inverse)
│       └── mel_band.py           # MelBandProjection (Mel-RoFormer band split)
├── external/                     # reference repos (BS-RoFormer, GuideSep)
├── papers/                       # reference papers
└── requirements.txt
```

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt          # CPU torch: add --index-url https://download.pytorch.org/whl/cpu
```

## Run the smoke test

```bash
.venv/bin/python scripts/test_pipeline.py
```

Expected (abridged) output:

```
[1] Waveforms: mixture (48000,)  reference (188775,)  target (48000,)
[2] STFT of mixture: (1, 48000) -> (1, 513, 188, 2)
[3] Mel-band projection: (1, 513, 188, 2) -> (1, 188, 60, 384)
[4] iSTFT round-trip: RMSE vs input ≈ 1e-8   # lossless
```

## Key design notes

- **One config object** (`FeatureConfig`) feeds every stage so STFT/iSTFT/mel
  parameters can never drift apart.
- **`MelBandProjection` is a band *split*, not a mel spectrogram** — it groups
  raw complex FFT bins into overlapping mel bands and projects each with a
  per-band `RMSNorm → Linear`, preserving phase for later masking + iSTFT.
- **Manifest-driven dataset**: a list of `{mixture, reference, target}` paths,
  decoupling data location from code.
