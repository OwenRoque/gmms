"""Objetos de configuracion centralizados para el feature pipeline de GMSS.

Por que un solo config object?
------------------------------
Las etapas STFT / iSTFT / mel-band DEBEN compartir *exactamente* los mismos
parametros. Si, por ejemplo, el extractor usa ``n_fft=2048`` pero la mel
projection asume ``n_fft=1024``, el numero de frequency bins no coincide 
y la reconstruccion del waveform no sera correcta. 

Los defaults siguen el paper de Mel-RoFormer (44.1 kHz, n_fft=2048, hop=512).
Para reproducir el setup de GuideSep, construir
``FeatureConfig(sample_rate=16000, n_fft=1024, hop_length=256)`` (o cargarlo
desde un YAML).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureConfig:
    """Bundle inmutable de parametros de audio + STFT + mel-band.

    Attributes
    ----------
    sample_rate:
        Sample rate objetivo (Hz). Todo waveform se remuestrea a este valor para
        que el eje de frecuencia del STFT sea consistente en el dataset.
    mono:
        Si ``True`` hacemos downmix a un solo channel. El pipeline actual es
        solo mono por simplicidad; stereo queda como extension futura documentada.
    n_fft:
        Tamano de la FFT. Produce ``n_fft // 2 + 1`` frequency bins.
    hop_length:
        Samples entre frames STFT consecutivos. Hop mas chico -> mas time frames
        -> mejor resolucion temporal pero mas compute.
    win_length:
        Longitud de la ventana de analisis. Default a ``n_fft`` cuando es ``None``.
    num_bands:
        Numero de mel bands que usa :class:`MelBandProjection`. 60 es el valor
        de la implementacion de referencia de Mel-RoFormer.
    dim:
        Ancho de features al que se proyecta cada band (model dim del transformer).
    normalized:
        Se pasa directo a ``torch.stft`` / ``torch.istft``.
    """

    # --- audio ---
    sample_rate: int = 44100
    mono: bool = True

    # --- stft ---
    n_fft: int = 2048
    hop_length: int = 512
    win_length: int | None = None
    normalized: bool = False

    # --- mel-band projection ---
    num_bands: int = 60
    dim: int = 384

    @property
    def effective_win_length(self) -> int:
        """Longitud de ventana que se usa realmente (fallback a ``n_fft``)."""
        return self.win_length if self.win_length is not None else self.n_fft

    @property
    def num_freqs(self) -> int:
        """Numero de frequency bins que produce el STFT (one-sided)."""
        return self.n_fft // 2 + 1
