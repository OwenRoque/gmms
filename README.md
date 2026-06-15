# GMSS — Guided Music Source Separation

Guided source-separation system inspired by **Mel-Band RoFormer**, adapted to take a
**reference/condition** signal (e.g. a hummed melody) as in **User-Guided Generative
Source Separation**.

Built on [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template/)
(PyTorch Lightning + Hydra).

```
mixture waveform → STFT → MelBandProjection → MelBandMaskEstimator → iSTFT → separated
```

## Project structure

```
gmss/
├── .project-root
├── configs/
│   ├── test_separation.yaml
│   ├── data/guidesep_smoke.yaml
│   ├── model/separation_smoke.yaml
│   ├── trainer/cpu.yaml
│   ├── paths/default.yaml
│   ├── hydra/default.yaml
│   └── extras/default.yaml
├── src/
│   ├── test_separation.py          # Hydra entry point (smoke test)
│   ├── data/
│   │   ├── audio_loader.py
│   │   ├── dataset.py
│   │   └── guidesep_datamodule.py
│   ├── models/
│   │   ├── separation_smoke_module.py
│   │   └── components/
│   │       ├── feature_config.py
│   │       ├── minimal_separator.py
│   │       └── features/
│   └── utils/
├── external/GuideSep/              # reference audio (gitignored)
└── requirements.txt
```

## Setup (conda)

```bash
conda create -n gmss python=3.8
conda activate gmss

# install pytorch (>=2.0.1)
conda install pytorch torchvision torchaudio -c pytorch -c nvidia

pip install -r requirements.txt
```

On CPU-only machines, replace the conda line with:

```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

## Run the separation smoke test

```bash
conda activate gmss
python src/test_separation.py
```

Override config from CLI:

```bash
python src/test_separation.py data.segment_seconds=5.0
```

Expected metrics:

| Case | RMSE vs mixture |
|------|-----------------|
| `init_identity=True` | ≈ 1e-8 (output ≈ mixture) |
| `init_identity=False` | ~0.15 (random mask, end-to-end demo) |

## Key design notes

- **Hydra configs** compose data, model, and trainer; paths resolve via `${paths.root_dir}`.
- **`FeatureConfig`** is shared across STFT, mel-band split, and mask estimator.
- **`MelBandProjection`** is a band *split*, not a mel spectrogram — it preserves complex phase for iSTFT.
- **Manifest-driven dataset**: triplets `{mixture, reference, target}` decouple data paths from code.
