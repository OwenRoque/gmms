"""STFTFeatureExtractor: waveform <-> espectrograma complejo (como tensor real).

Notas de diseno
---------------
* Es un ``nn.Module`` (no una funcion suelta) para que la ventana de analisis sea
  un buffer registrado que siga automaticamente ``.to(device)`` / ``.cuda()`` y
  para poder embeber el extractor dentro de un model graph mas grande mas adelante.
* ``torch.stft`` devuelve un tensor *complejo* ``[F, T]``. Autograd sobre ops
  complejas es incomodo, asi que -- igual que Mel-RoFormer -- hacemos
  ``view_as_real`` de inmediato para obtener ``[F, T, 2]`` (ultimo eje = [real, imag]).
* Forward (``stft``) e inverse (``istft``) viven en el mismo modulo para que sus
  parametros no puedan desincronizarse, garantizando espectrogramas reconstruibles.
"""

from __future__ import annotations

import torch
from torch import nn

from gmss.config import FeatureConfig


class STFTFeatureExtractor(nn.Module):
    """Calcula el STFT (one-sided) y su inversa con parametros compartidos.

    Shapes (input mono)
    --------------------
    ``stft``:   ``[B, T_samples]``           -> ``[B, F, T_frames, 2]``
    ``istft``:  ``[B, F, T_frames, 2]``      -> ``[B, T_samples]``

    donde ``F = n_fft // 2 + 1`` y el ultimo eje del espectrograma apila
    las partes real e imaginaria.
    """

    def __init__(self, config: FeatureConfig) -> None:
        super().__init__()
        self.config = config
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.win_length = config.effective_win_length
        self.normalized = config.normalized

        # Ventana Hann como buffer (non-persistent): se mueve con el modulo entre
        # devices pero no se guarda en state_dict (queda determinada por win_length).
        window = torch.hann_window(self.win_length)
        self.register_buffer("window", window, persistent=False)

    def stft(self, waveform: torch.Tensor) -> torch.Tensor:
        """STFT forward -> espectrograma real-valued ``[B, F, T, 2]``."""
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)  # [T] -> [1, T]

        complex_spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            normalized=self.normalized,
            return_complex=True,
        )  # [B, F, T]

        # [B, F, T] complex -> [B, F, T, 2] real.
        return torch.view_as_real(complex_spec)

    def istft(
        self,
        spectrogram: torch.Tensor,
        length: int | None = None,
    ) -> torch.Tensor:
        """STFT inversa. Input ``[B, F, T, 2]`` -> waveform ``[B, T_samples]``.

        Parameters
        ----------
        length:
            Si se pasa, la salida se recorta/rellena a exactamente esa cantidad
            de samples (el framing del STFT puede cambiar la longitud ligeramente).
        """
        complex_spec = torch.view_as_complex(spectrogram.contiguous())
        return torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            normalized=self.normalized,
            length=length,
            return_complex=False,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Alias de :meth:`stft` para usar el modulo como callable."""
        return self.stft(waveform)
