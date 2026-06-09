"""MelBandProjection: band split estilo Mel-RoFormer (NO es un mel spectrogram).

La idea clave (leer esto!)
--------------------------
Un mel spectrogram *clasico* multiplica el power spectrum por filtros mel
triangulares y suma, colapsando cada band a un solo valor de energia. Eso
descarta la fase y es irreversible -- inutil si despues queremos enmascarar el
espectro complejo y correr un iSTFT.

Mel-RoFormer en cambio usa el mel filter bank solo para *agrupar* los FFT bins crudos:

    1. Construir un mel filter bank con librosa -> shape [num_bands, num_freqs].
    2. Tratar ``filter > 0`` como mapa binario de membresia: que FFT bins
       pertenecen a que band. Las bands se solapan, y como el espaciado mel es
       mas denso en bajas frecuencias, las bands bajas tienen pocos bins y las
       altas tienen muchos.
    3. Por cada band, reunir sus bins complejos crudos (real+imag => ``2 * n_bins``
       numeros reales) y proyectarlos con un ``RMSNorm -> Linear`` pequeño a un
       ancho uniforme ``dim``.

Resultado: ``[B, T, num_bands, dim]`` -- secuencia de tokens que consume el
RoFormer, con *toda* la info compleja preservada para mask estimation mas adelante.
"""

from __future__ import annotations

import librosa
import torch
from torch import nn

from gmss.config import FeatureConfig


class RMSNorm(nn.Module):
    """Variante Root-mean-square LayerNorm usada en modelos RoFormer.

    Mas barata que LayerNorm (no resta la media) y empiricamente estable para
    transformers. ``scale = sqrt(dim)`` mantiene la magnitud post-norm razonable.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(x, dim=-1) * self.scale * self.gamma


class MelBandProjection(nn.Module):
    """Agrupa STFT bins crudos en mel bands solapadas y proyecta cada una a ``dim``.

    Parameters
    ----------
    config:
        :class:`FeatureConfig` compartido (usa ``sample_rate``, ``n_fft``,
        ``num_bands`` y ``dim``).

    Input / output shapes (mono)
    ----------------------------
    forward:  ``[B, F, T, 2]``  ->  ``[B, T, num_bands, dim]``
    """

    def __init__(self, config: FeatureConfig) -> None:
        super().__init__()
        self.config = config
        self.num_bands = config.num_bands
        self.num_freqs = config.num_freqs

        # Construir el mel filter bank
        mel_fb_np = librosa.filters.mel(
            sr=config.sample_rate, n_fft=config.n_fft, n_mels=config.num_bands
        )  # [num_bands, num_freqs]
        mel_fb = torch.from_numpy(mel_fb_np)

        # Sugerencia de @firebirdblue23: el primer y ultimo bin
        # pueden quedar sin cubrir por los filtros triangulares de librosa;
        # empujarlos para que toda frecuencia pertenezca al menos a una band.
        mel_fb[0, 0] = mel_fb[0, 1] * 0.25
        mel_fb[-1, -1] = mel_fb[-1, -2] * 0.25

        # Membresia binaria: que bins pertenecen a que band.
        freqs_per_band = mel_fb > 0  # bool [num_bands, num_freqs]
        assert freqs_per_band.any(dim=0).all(), (
            "every frequency bin must be covered by at least one band"
        )

        # Por band, cuantos bins le corresponden. Tuple de ints porque lo
        # necesitamos para dimensionar los Linear por band de abajo.
        num_freqs_per_band = freqs_per_band.sum(dim=-1)  # [num_bands]
        self.num_freqs_per_band: tuple[int, ...] = tuple(
            num_freqs_per_band.tolist()
        )

        # Indice flat de gather: permite extraer los bins de todas las
        # bands del espectro en una sola operacion de advanced indexing.
        band_ids, freq_ids = torch.where(freqs_per_band)  # ambos [total_bins]
        # Iteramos bands en orden ascendente, asi que ordenar por band y luego freq.
        order = torch.argsort(band_ids * self.num_freqs + freq_ids)
        freq_indices = freq_ids[order]  # [total_bins]
        self.register_buffer("freq_indices", freq_indices, persistent=False)

        # Cuantas bands cubre cada frecuencia (se usa despues al promediar
        # masks solapadas; guardado ahora para uso downstream).
        num_bands_per_freq = freqs_per_band.sum(dim=0)  # [num_freqs]
        self.register_buffer(
            "num_bands_per_freq", num_bands_per_freq, persistent=False
        )

        # Una projection head pequeña por band. Input width = ``2 * n_bins`` 
        # porque cada bin complejo aporta (real, imag).
        self.to_features = nn.ModuleList(
            [
                nn.Sequential(RMSNorm(2 * n_bins), nn.Linear(2 * n_bins, config.dim))
                for n_bins in self.num_freqs_per_band
            ]
        )

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """``[B, F, T, 2]`` -> ``[B, T, num_bands, dim]``."""
        batch = spectrogram.shape[0]

        # Reunir los bins de todas las bands, en orden ascendente de band:
        # [B, F, T, 2] -> index dim 1 -> [B, total_bins, T, 2]
        gathered = spectrogram[:, self.freq_indices]

        # Mover time al frente de los ejes de features y aplanar (bins, complex)
        # en una sola dim de features: [B, total_bins, T, 2] -> [B, T, total_bins*2]
        gathered = gathered.permute(0, 2, 1, 3).reshape(batch, -1, self._total_flat)

        # Partir la dim flat de vuelta en chunks por band de tamano 2*n_bins
        # y proyectar cada chunk de forma independiente.
        split_sizes = [2 * n for n in self.num_freqs_per_band]
        chunks = gathered.split(split_sizes, dim=-1)

        band_features = [proj(chunk) for proj, chunk in zip(self.to_features, chunks)]

        # Apilar en un nuevo eje de band: list de [B, T, dim] -> [B, T, num_bands, dim]
        return torch.stack(band_features, dim=-2)

    @property
    def total_bins(self) -> int:
        """Numero de entradas en el indice flat de gather (con solapamiento)."""
        return int(self.freq_indices.shape[0])

    @property
    def _total_flat(self) -> int:
        return 2 * self.total_bins


class MelBandMaskEstimator(nn.Module):
    """Proyecta band features de vuelta a mascara compleja por bin (inverso del band split).

    Una head pequena por band: ``RMSNorm -> Linear`` -> ``2 * n_bins`` (pares real/imag).
    Sin backbone intermedio en esta v1: las features entran directo al estimator.

    Parameters
    ----------
    config:
        :class:`FeatureConfig` compartido.
    band_split:
        :class:`MelBandProjection` con el que comparte layout de bands e indices.
    init_identity:
        Si ``True``, inicializa biases para mascara ~1+0j (output ~= mixture).
    """

    def __init__(
        self,
        config: FeatureConfig,
        band_split: MelBandProjection,
        init_identity: bool = True,
    ) -> None:
        super().__init__()
        if band_split.num_bands != config.num_bands:
            raise ValueError("band_split.num_bands must match config.num_bands")

        self.config = config
        self.band_split = band_split

        self.to_masks = nn.ModuleList(
            [
                nn.Sequential(
                    RMSNorm(config.dim),
                    nn.Linear(config.dim, 2 * n_bins),
                )
                for n_bins in band_split.num_freqs_per_band
            ]
        )

        if init_identity:
            self._init_identity_masks()

    def _init_identity_masks(self) -> None:
        """Mascara compleja identidad (1+0j) por bin -> separated ~= mixture sin entrenar."""
        with torch.no_grad():
            for head in self.to_masks:
                linear = head[-1]
                assert isinstance(linear, nn.Linear)
                linear.weight.zero_()
                bias = linear.bias.view(-1, 2)
                bias[:, 0] = 1.0  # real
                bias[:, 1] = 0.0  # imag

    def forward(self, band_features: torch.Tensor) -> torch.Tensor:
        """``[B, T, num_bands, dim]`` -> ``[B, T, total_flat]``."""
        bands = band_features.unbind(dim=-2)
        chunks = [head(feats) for head, feats in zip(self.to_masks, bands)]
        return torch.cat(chunks, dim=-1)


def apply_mel_band_masks(
    spectrogram: torch.Tensor,
    mask_flat: torch.Tensor,
    band_split: MelBandProjection,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Aplicar mascaras por band al espectrograma, promediando en bins solapados.

    Sigue la logica de Mel-RoFormer: ``scatter_add`` sobre frequency bins y division
    por ``num_bands_per_freq``.

    Parameters
    ----------
    spectrogram:
        Espectrograma de la mixture ``[B, F, T, 2]``.
    mask_flat:
        Mascaras aplanadas ``[B, T, total_flat]`` (pares real/imag por bin gathered).
    band_split:
        Modulo con ``freq_indices`` y ``num_bands_per_freq``.

    Returns
    -------
    torch.Tensor
        Espectrograma enmascarado ``[B, F, T, 2]``.
    """
    batch, num_freqs, num_frames, _ = spectrogram.shape
    total_bins = band_split.total_bins

    if mask_flat.shape[-1] != 2 * total_bins:
        raise ValueError(
            f"mask_flat last dim {mask_flat.shape[-1]} != 2 * total_bins ({2 * total_bins})"
        )

    masks = mask_flat.reshape(batch, num_frames, total_bins, 2)
    masks_c = torch.view_as_complex(masks.contiguous())  # [B, T, total_bins]
    spec_c = torch.view_as_complex(spectrogram.contiguous())  # [B, F, T]

    masks_summed = torch.zeros(
        batch, num_freqs, num_frames,
        dtype=masks_c.dtype,
        device=masks_c.device,
    )

    freq_idx = band_split.freq_indices.view(1, -1, 1).expand(batch, -1, num_frames)
    masks_summed.scatter_add_(1, freq_idx, masks_c.permute(0, 2, 1))

    denom = band_split.num_bands_per_freq.clamp(min=eps).view(1, num_freqs, 1)
    masks_avg = masks_summed / denom.to(masks_summed.dtype)

    masked = spec_c * masks_avg
    return torch.view_as_real(masked)
