"""Smoke test: pipeline de separacion v1 (mixture -> mask -> iSTFT).

Ejecutar desde la raiz del proyecto:

    .venv/bin/python scripts/test_separation.py

Con ``init_identity=True`` (default) la salida debe ser casi identica a la mixture
(RMSE muy bajo). Con ``init_identity=False`` la salida es ruidosa pero el pipeline
sigue siendo end-to-end.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gmss.config import FeatureConfig
from gmss.data import AudioLoader, GuidedSeparationDataset, Triplet
from gmss.models import MinimalSeparator


def build_example_triplet() -> Triplet:
    guidesep = PROJECT_ROOT / "external" / "GuideSep"
    mixture = guidesep / "mix" / "real_3_mix.wav"
    reference = guidesep / "cond" / "real_3_humming.wav"
    target = mixture
    return Triplet(mixture=str(mixture), reference=str(reference), target=str(target))


def run_case(label: str, separator: MinimalSeparator, mixture: torch.Tensor) -> None:
    print(f"\n--- {label} ---")
    separated, masked_spec, mask_flat = separator(
        mixture, output_length=mixture.shape[-1], return_spec=True
    )
    rmse = torch.mean((separated - mixture) ** 2).sqrt().item()
    print(f"    mixture   {tuple(mixture.shape)}")
    print(f"    separated {tuple(separated.shape)}")
    print(f"    masked_spec {tuple(masked_spec.shape)}")
    print(f"    mask_flat   {tuple(mask_flat.shape)}")
    print(f"    RMSE vs mixture = {rmse:.2e}")


def main() -> None:
    config = FeatureConfig(
        sample_rate=16000,
        n_fft=1024,
        hop_length=256,
        num_bands=60,
        dim=384,
    )

    print("=" * 70)
    print("GMSS minimal separation pipeline v1")
    print("=" * 70)

    loader = AudioLoader(sample_rate=config.sample_rate, mono=config.mono)
    dataset = GuidedSeparationDataset(
        triplets=[build_example_triplet()],
        loader=loader,
        segment_seconds=3.0,
        train=True,
        seed=0,
    )
    mixture = dataset[0]["mixture"].unsqueeze(0)

    identity_sep = MinimalSeparator(config, init_identity_mask=True)
    run_case("init_identity=True (esperado ~= mixture)", identity_sep, mixture)

    random_sep = MinimalSeparator(config, init_identity_mask=False)
    run_case("init_identity=False (mascara aleatoria, demo end-to-end)", random_sep, mixture)

    print("\nPipeline de separacion v1 OK.")


if __name__ == "__main__":
    main()
