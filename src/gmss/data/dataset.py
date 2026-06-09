"""GuidedSeparationDataset: dataset minimo para guided MSS.

Siguiendo el setup de User-Guided Generative Source Separation, cada ejemplo
de entrenamiento es un ``triplet``:

    mixture    -- la mezcla completa que queremos separar
    reference  -- senal de guia/condicion (ej. melodía tarareada) que le dice
                  al modelo *que* fuente extraer
    target     -- el stem aislado ground-truth (para la loss mas adelante)

Por que manifest-driven?
------------------------
TODO: Usar Lightning Hydra Template (Pytorch Lightning, Hydra) para datasets y otros
 (tasks, experimentos, etc.)   

El dataset recibe una lista de dicts ``{"mixture", "reference", "target"}`` con
rutas, en lugar de recorrer un directorio. Esto desacopla donde viven los
archivos de lo que hace el dataset, es trivialmente reproducible (el manifest
es solo data), y permite mezclar fuentes sin tocar codigo. Un manifest JSON se
carga con :meth:`from_manifest`.

Es intencionalmente minimo: devuelve solo waveforms (feature extraction vive en
el model / collate stage), soporta segmentos de longitud fija para que un
DataLoader pueda hacer batch, y no contiene augmentation ni logica de training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import torch
from torch.utils.data import Dataset

from gmss.data.audio_loader import AudioLoader


class Triplet(TypedDict):
    """Almacena las rutas de los tres waveforms (mixture, reference, target)."""

    mixture: str
    reference: str
    target: str


class Sample(TypedDict):
    """Lo que devuelve ``__getitem__``: los tres tensores de waveform."""

    mixture: torch.Tensor
    reference: torch.Tensor
    target: torch.Tensor


class GuidedSeparationDataset(Dataset[Sample]):
    """Devuelve waveforms (mixture, reference, target) para guided separation.

    Parameters
    ----------
    triplets:
        Lista de dicts con rutas ``mixture``/``reference``/``target``.
    loader:
        Un :class:`AudioLoader`. Compartir un loader en todo el dataset reutiliza
        los resamplers en cache.
    segment_seconds:
        Si se define, mixture y target se recortan a esta duracion (random crop
        cuando ``train=True``, sino crop deterministico desde el inicio). La
        referencia queda a longitud completa porque un hum de guia no necesita
        estar alineado en tiempo con la mixture. ``None`` devuelve clips completos.
    train:
        Controla random vs deterministic cropping.
    seed:
        Seed base para random crops reproducibles.
    """

    def __init__(
        self,
        triplets: list[Triplet],
        loader: AudioLoader,
        segment_seconds: float | None = None,
        train: bool = True,
        seed: int = 0,
    ) -> None:
        self.triplets = triplets
        self.loader = loader
        self.segment_seconds = segment_seconds
        self.train = train
        self.seed = seed

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        loader: AudioLoader,
        **kwargs: object,
    ) -> "GuidedSeparationDataset":
        """Construir un dataset desde un JSON con una lista de triplets."""
        with open(manifest_path, "r", encoding="utf-8") as f:
            triplets: list[Triplet] = json.load(f)
        return cls(triplets, loader, **kwargs)  # type: ignore[arg-type]

    def __len__(self) -> int:
        return len(self.triplets)

    @property
    def _segment_samples(self) -> int | None:
        """Devuelve el numero de muestras de un segmento."""
        if self.segment_seconds is None:
            return None
        return int(self.segment_seconds * self.loader.sample_rate)

    def _crop(self, waveform: torch.Tensor, index: int) -> torch.Tensor:
        """Recortar/rellenar un waveform 1-D a ``_segment_samples``."""
        num_samples = self._segment_samples
        # Sin segmentacion
        if num_samples is None:
            return waveform

        length = waveform.shape[-1]
        # Si el waveform es mas corto que el segmento
        if length < num_samples:
            # Right-pad con ceros para que clips cortos sigan produciendo un segmento completo. (evitar este caso)
            pad = num_samples - length
            return torch.nn.functional.pad(waveform, (0, pad))

        # Mismo tamano
        if length == num_samples:
            return waveform

        # Si es training, usar random crop.
        if self.train:
            # Randomness reproducible por item: epocas deterministicas dado el seed
            # (ayuda al debugging).
            generator = torch.Generator().manual_seed(self.seed + index)
            start = int(
                torch.randint(0, length - num_samples + 1, (1,), generator=generator)
            )
        else:
            start = 0
        return waveform[..., start : start + num_samples]

    def __getitem__(self, index: int) -> Sample:
        triplet = self.triplets[index]

        mixture = self.loader.load(triplet["mixture"])
        reference = self.loader.load(triplet["reference"])
        target = self.loader.load(triplet["target"])

        mixture = self._crop(mixture, index)
        target = self._crop(target, index)
        # Reference queda a longitud completa

        return Sample(mixture=mixture, reference=reference, target=target)
