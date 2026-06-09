"""AudioLoader: carga robusta de waveforms basada en torchaudio.

Responsabilidades:
--------------------
    1. Cargar un archivo de audio a un tensor float ``[channels, samples]``.
    2. Opcionalmente convertir a mono (solo para pruebas).
    3. Remuestrear al sample rate objetivo.
    4. Normalizar por pico para evitar clipping. (Mas adelante evaluar uso con medidas como RMS o LUFS)
    
Para la decodificacion de archivos se usa ``soundfile`` (libsndfile): versiones recientes de
``torchaudio`` delegan ``torchaudio.load`` a un backend opcional ``torchcodec``/ffmpeg,
así que leer via ``soundfile`` mantiene el loader liviano en dependencias y portable.
Todo el DSP posterior (resampling) queda en ``torchaudio`` para que los datos sigan siendo tensores torch.
"""


from __future__ import annotations

from pathlib import Path

import soundfile as sf
import torch
import torchaudio


class AudioLoader:
    """Carga y estandariza audio a un tensor 1-D (mono) o 2-D (multi-channel).

    Parameters
    ----------
    sample_rate:
        Tasa objetivo (Hz). Todos los clips quedan a esta tasa.
    mono:
        Si es ``True``, promediar canales a un solo canal y devolver un
        tensor 1-D ``[samples]``. Si es ``False`` devolver ``[channels, samples]``.
    normalize:
        Si es ``True``, dividir por el pico absoluto (peak normalization) para que el sample mas
        fuerte sea +/-1. Esto elimina diferencias de ganancia arbitrarias entre archivos.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        mono: bool = True,
        normalize: bool = True,
    ) -> None:
        self.sample_rate = sample_rate
        self.mono = mono
        self.normalize = normalize
        # Guardar en cache una transformacion Resample por sample-rate de origen.
        # Construir un resampler es relativamente costoso (precalcula un kernel de filtro)
        # Solucion: reutilizar entre muchos archivos que comparten la misma tasa de origen.

        self._resamplers: dict[int, torchaudio.transforms.Resample] = {}

    def _get_resampler(self, orig_sr: int) -> torchaudio.transforms.Resample:
        """Devolver el resampler correspondiente a la sample-rate de origen."""
        # Si el resampler no esta en cache, construirlo.
        if orig_sr not in self._resamplers:
            self._resamplers[orig_sr] = torchaudio.transforms.Resample(
                orig_freq=orig_sr, new_freq=self.sample_rate
            )
        return self._resamplers[orig_sr]

    @staticmethod
    def _to_mono(waveform: torch.Tensor) -> torch.Tensor:
        """ Promediar a lo largo de la dimension del channel ``[C, T] -> [1, T]``."""
        # Si el waveform ya es mono, devolverlo.
        if waveform.shape[0] == 1:
            return waveform
        # Estereo o multi-channel -> promediar
        return waveform.mean(dim=0, keepdim=True)

    def _resample(self, waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """Remuestrear el waveform a la sample-rate objetivo."""
        # Sin cambios si las tasas coinciden
        if orig_sr == self.sample_rate:
            return waveform
        # Sino, usar/crear el resampler correspondiente.
        return self._get_resampler(orig_sr)(waveform)

    @staticmethod
    def _peak_normalize(waveform: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        peak = waveform.abs().max()
        # Clip silencioso, no hacer nada.
        if peak < eps:  
            return waveform
        # Normalizar por el pico absoluto.
        return waveform / peak 

    def load(self, path: str | Path) -> torch.Tensor:
        """Carga un archivo y devuelve un waveform estandarizado float32.

        Devuelve
        -------
        torch.Tensor
            ``[samples]`` si ``mono`` es ``True`` sino ``[channels, samples]``.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")

        # soundfile devuelve (frames, channels) -> transponer a [channels, frames].
        data, orig_sr = sf.read(str(path), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T).contiguous()  # [channels, samples]

        if self.mono:
            waveform = self._to_mono(waveform)

        waveform = self._resample(waveform, orig_sr)

        if self.normalize:
            waveform = self._peak_normalize(waveform)

        if self.mono:
            # Eliminar canal redundante -> [samples].
            waveform = waveform.squeeze(0)

        return waveform.contiguous()
