"""GuidedCrossAttention: condicionamiento por referencia previo al backbone Mel-RoFormer.

Inserta informacion de una senal de referencia dentro de la representacion
band-split de la mezcla mediante Multi-Head Cross-Attention:

    Query  = features de la mezcla    (B, T, F, D)
    Key    = features de la referencia (B, S, F, D)
    Value  = features de la referencia (B, S, F, D)

La atencion se realiza *alineada por banda mel*: la banda f de la mezcla atiende
unicamente a la banda f de la referencia a lo largo del tiempo. Esto mantiene la
complejidad lineal en el numero de bandas y es coherente con la naturaleza axial
de Mel-RoFormer.

La salida conserva la forma exacta de la mezcla (B, T, F, D), por lo que el modulo
es drop-in entre MelBandProjection y el backbone, sin capas adaptadoras.

Ninguna dimension es fija: todo se deriva de FeatureConfig o del tensor de entrada.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from typing import Optional

from src.models.components.feature_config import FeatureConfig
from src.models.components.sdp_attention import sdp_attention


class GuidedCrossAttention(nn.Module):
    """Multi-Head Cross-Attention guiado por referencia, alineado por banda mel."""

    def __init__(
        self,
        config: FeatureConfig,
        heads: Optional[int] = None,
        dropout: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.config = config

        self.dim = config.dim
        self.heads = heads if heads is not None else config.cross_attn_heads
        dropout = dropout if dropout is not None else config.cross_attn_dropout

        if self.dim % self.heads != 0:
            raise ValueError(
                f"dim ({self.dim}) debe ser divisible por heads ({self.heads})"
            )
        self.head_dim = self.dim // self.heads
        self.scale = self.head_dim**-0.5
        self.attn_dropout = dropout
        self.flash_attn = config.flash_attn

        # Pre-LayerNorm independiente para query (mezcla) y key/value (referencia).
        self.norm_query = nn.LayerNorm(self.dim)
        self.norm_context = nn.LayerNorm(self.dim)

        # Proyecciones lineales D -> D (sin bias, estilo RoFormer).
        self.to_q = nn.Linear(self.dim, self.dim, bias=False)
        self.to_k = nn.Linear(self.dim, self.dim, bias=False)
        self.to_v = nn.Linear(self.dim, self.dim, bias=False)
        self.to_out = nn.Linear(self.dim, self.dim, bias=False)

        self.out_dropout = nn.Dropout(dropout)

    def forward(
        self,
        mixture_features: torch.Tensor,
        reference_features: torch.Tensor,
    ) -> torch.Tensor:
        """Condiciona la mezcla con la referencia.

        Args:
            mixture_features:   (B, T, F, D) features band-split de la mezcla.
            reference_features: (B, S, F, D) features band-split de la referencia.

        Returns:
            (B, T, F, D) mezcla condicionada (misma forma que ``mixture_features``).
        """
        if mixture_features.ndim != 4 or reference_features.ndim != 4:
            raise ValueError(
                "Se esperan tensores 4D (B, T, F, D); "
                f"recibido mix={tuple(mixture_features.shape)}, "
                f"ref={tuple(reference_features.shape)}"
            )

        b, t, f, d = mixture_features.shape
        bf = b * f

        if reference_features.shape[0] != b:
            raise ValueError("mezcla y referencia deben compartir batch size")
        if reference_features.shape[2] != f:
            raise ValueError("mezcla y referencia deben compartir num_bands (F)")
        if d != self.dim:
            raise ValueError(f"dim de entrada ({d}) != config.dim ({self.dim})")

        residual = mixture_features

        # Pre-norm.
        q_in = self.norm_query(mixture_features)
        kv_in = self.norm_context(reference_features)

        # Banda como dimension de batch: (B, *, F, D) -> (B*F, *, D).
        q = rearrange(q_in, "b t f d -> (b f) t d")
        k = rearrange(kv_in, "b s f d -> (b f) s d")
        v = k

        # Proyecciones Q, K, V.
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        # Split en heads: (B*F, L, D) -> (B*F, H, L, E).
        q = rearrange(q, "n t (h e) -> n h t e", h=self.heads)
        k = rearrange(k, "n s (h e) -> n h s e", h=self.heads)
        v = rearrange(v, "n s (h e) -> n h s e", h=self.heads)

        # Scaled dot-product attention (kernel flash / memory-efficient si esta disponible).
        # scores implicitos: (B*F, H, T, S), softmax sobre S.
        attn_out = sdp_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout if self.training else 0.0,
            flash=self.flash_attn,
        )

        # Merge heads: (B*F, H, T, E) -> (B*F, T, D).
        attn_out = rearrange(attn_out, "n h t e -> n t (h e)")

        # Output projection.
        attn_out = self.to_out(attn_out)

        # De vuelta a (B, T, F, D).
        attn_out = rearrange(attn_out, "(b f) t d -> b t f d", b=b, f=f)

        # Conexion residual sobre la mezcla.
        return residual + self.out_dropout(attn_out)
