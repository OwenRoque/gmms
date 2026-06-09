"""Smoke test: cargar UN sample de guided separation e imprimir cada tensor shape.

Ejecutar desde la raiz del proyecto:

    .venv/bin/python scripts/test_pipeline.py

Usa el audio de ejemplo que trae la referencia GuideSep
(``external/GuideSep``): la mixture y una condicion tarareada. No hay stem
aislado ground-truth en ese ejemplo, asi que para este test de *shapes*
reutilizamos la mixture como target placeholder (en training real el target
es la fuente aislada).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Permitir ``import gmss`` sin instalar el paquete.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gmss.config import FeatureConfig
from gmss.data import AudioLoader, GuidedSeparationDataset, Triplet
from gmss.features import MelBandProjection, STFTFeatureExtractor


def build_example_triplet() -> Triplet:
    """Apuntar a los clips de ejemplo de GuideSep (mix + humming condition)."""
    guidesep = PROJECT_ROOT / "external" / "GuideSep"
    mixture = guidesep / "mix" / "real_3_mix.wav"
    reference = guidesep / "cond" / "real_3_humming.wav"
    # Target placeholder (el ejemplo no trae stem aislado).
    target = mixture
    return Triplet(mixture=str(mixture), reference=str(reference), target=str(target))


def main() -> None:
    # Usar el sample rate de GuideSep para que los archivos de ejemplo encajen;
    # n_fft mas chico mantiene el printout legible.
    config = FeatureConfig(
        sample_rate=16000,
        n_fft=1024,
        hop_length=256,
        num_bands=60,
        dim=384,
    )

    print("=" * 70)
    print("GMSS feature-pipeline smoke test")
    print("=" * 70)
    print(f"sample_rate={config.sample_rate}  n_fft={config.n_fft}  "
          f"hop={config.hop_length}  num_freqs={config.num_freqs}  "
          f"num_bands={config.num_bands}  dim={config.dim}")

    # --- 1. Dataset devuelve los tres waveforms --------------------------------
    loader = AudioLoader(sample_rate=config.sample_rate, mono=config.mono)
    dataset = GuidedSeparationDataset(
        triplets=[build_example_triplet()],
        loader=loader,
        segment_seconds=3.0,
        train=True,
        seed=0,
    )
    print(f"\nDataset length: {len(dataset)}")

    sample = dataset[0]
    print("\n[1] Waveforms from dataset[0]:")
    for name, wav in sample.items():
        secs = wav.shape[-1] / config.sample_rate
        print(f"    {name:<10} shape={tuple(wav.shape)}  ({secs:.2f}s)  dtype={wav.dtype}")

    # --- 2. STFT ---------------------------------------------------------------
    extractor = STFTFeatureExtractor(config)
    # Agregar batch dim: [T] -> [1, T]
    mixture = sample["mixture"].unsqueeze(0)
    spec = extractor(mixture)  # [B, F, T, 2]
    print("\n[2] STFT of mixture:")
    print(f"    input  {tuple(mixture.shape)}  ->  spectrogram {tuple(spec.shape)}")
    print("    (last axis = [real, imag])")

    # --- 3. Mel-band projection ------------------------------------------------
    projection = MelBandProjection(config)
    band_features = projection(spec)  # [B, T, num_bands, dim]
    print("\n[3] Mel-band projection:")
    print(f"    spectrogram {tuple(spec.shape)}  ->  band features {tuple(band_features.shape)}")
    print(f"    bins per band (first 10): {projection.num_freqs_per_band[:10]} ...")
    print(f"    total gathered bins: {int(projection.freq_indices.shape[0])}")

    # --- 4. iSTFT round-trip sanity check --------------------------------------
    recon = extractor.istft(spec, length=mixture.shape[-1])  # [B, T]
    err = torch.mean((recon - mixture) ** 2).sqrt().item()
    print("\n[4] iSTFT round-trip:")
    print(f"    reconstructed {tuple(recon.shape)}  RMSE vs input = {err:.2e}")

    print("\nAll stages ran successfully.")


if __name__ == "__main__":
    main()
