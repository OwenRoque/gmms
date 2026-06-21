"""Bloques de construccion del backbone Mel-RoFormer para GMSS.

Adaptados de lucidrains/BS-RoFormer con los siguientes cambios:
  - Se elimina la logica multi-stem (num_stems, mask_estimators ModuleList).
  - Se elimina la dependencia de bs_roformer.attend: se usa
    F.scaled_dot_product_attention (flash attn automatico en PyTorch >= 2.0).
  - Se elimina hyper_connections: residuales estandar para evitar la
    complejidad de la dimension de streams en tensores 4D.
  - Se elimina PoPE: solo RoPE.
  - RoFormerBackbone encapsula el bucle axial (linear + time + freq).
  - MaskEstimator incluye MLP profundo + GLU (igual al original).

Dependencias: rotary_embedding_torch, einops, beartype.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from rotary_embedding_torch import RotaryEmbedding
from torch import nn
from typing import Optional, Tuple

from src.models.components.feature_config import FeatureConfig
from src.models.components.sdp_attention import sdp_attention


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _exists(val: object) -> bool:
    return val is not None


def _default(val: object, d: object) -> object:
    return val if _exists(val) else d


# ---------------------------------------------------------------------------
# Norma
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1) * self.scale * self.gamma


# ---------------------------------------------------------------------------
# Feed-Forward
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Attention (full, con RoPE opcional)
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """Multi-head self-attention con RoPE opcional y gated output.

    Retorna ``(output, orig_values)`` para el mecanismo de value residual.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        rotary_embed: Optional[RotaryEmbedding] = None,
        flash: bool = True,
        add_value_residual: bool = False,
    ) -> None:
        super().__init__()
        self.heads = heads
        dim_inner = heads * dim_head
        self.rotary_embed = rotary_embed
        self._flash = flash
        self._dropout = dropout

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)
        self.to_gates = nn.Linear(dim, heads)
        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias=False),
            nn.Dropout(dropout),
        )

        if add_value_residual:
            self.learned_value_residual_mix: Optional[nn.Module] = nn.Sequential(
                nn.Linear(dim, heads),
                Rearrange("b n h -> b h n 1"),
                nn.Sigmoid(),
            )
        else:
            self.learned_value_residual_mix = None

    def forward(
        self,
        x: torch.Tensor,
        value_residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_norm = self.norm(x)
        q, k, v = rearrange(
            self.to_qkv(x_norm), "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads
        )
        orig_v = v

        if self.learned_value_residual_mix is not None and value_residual is not None:
            mix = self.learned_value_residual_mix(x_norm)   # (b, h, n, 1)
            # lerp(end=mix, weight=value_residual): v + vr*(mix - v)
            v = v.lerp(mix, value_residual)

        if self.rotary_embed is not None:
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)

        out = sdp_attention(
            q, k, v,
            dropout_p=self._dropout if self.training else 0.0,
            flash=self._flash,
        )

        gates = self.to_gates(x_norm)
        out = out * rearrange(gates, "b n h -> b h n 1").sigmoid()
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out), orig_v


# ---------------------------------------------------------------------------
# Linear Attention
# ---------------------------------------------------------------------------

class LinearAttention(nn.Module):
    """Atencion lineal O(n) — El-Nouby et al. 2021."""

    def __init__(
        self,
        dim: int,
        dim_head: int = 32,
        heads: int = 8,
        dropout: float = 0.0,
        add_value_residual: bool = False,
    ) -> None:
        super().__init__()
        dim_inner = dim_head * heads
        self.heads = heads
        self._dropout = dropout

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias=False),
            Rearrange("b n (qkv h d) -> qkv b h d n", qkv=3, h=heads),
        )
        self.temperature = nn.Parameter(torch.zeros(heads, 1, 1))
        self.to_out = nn.Sequential(
            Rearrange("b h d n -> b n (h d)"),
            nn.Linear(dim_inner, dim, bias=False),
        )

        if add_value_residual:
            self.learned_value_residual_mix: Optional[nn.Module] = nn.Sequential(
                nn.Linear(dim, heads),
                Rearrange("b n h -> b h 1 n"),
                nn.Sigmoid(),
            )
        else:
            self.learned_value_residual_mix = None

    def forward(
        self,
        x: torch.Tensor,
        value_residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_norm = self.norm(x)
        q, k, v = self.to_qkv(x_norm)
        orig_v = v

        if self.learned_value_residual_mix is not None and value_residual is not None:
            mix = self.learned_value_residual_mix(x_norm)   # (b, h, 1, n)
            v = v.lerp(value_residual, mix)

        q = F.normalize(q, dim=-2)
        k = F.normalize(k, dim=-2)
        q = q * self.temperature.exp()

        # Atencion lineal: K^T V luego Q (K^T V)
        # Escala manual para estabilidad
        kv = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d n, b h d e -> b h e n", q, kv)

        return self.to_out(out), orig_v


# ---------------------------------------------------------------------------
# Transformer (residuales estandar, sin hyper_connections)
# ---------------------------------------------------------------------------

class Transformer(nn.Module):
    """Pila de bloques attention + FF con pre-norm y residuales estandar.

    Retorna ``(output, first_values)`` para propagar value residual entre capas
    del backbone axial.
    """

    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        dim_head: int = 64,
        heads: int = 8,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        ff_mult: int = 4,
        norm_output: bool = True,
        rotary_embed: Optional[RotaryEmbedding] = None,
        flash_attn: bool = True,
        linear_attn: bool = False,
        add_value_residual: bool = False,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()

        for _ in range(depth):
            if linear_attn:
                attn: nn.Module = LinearAttention(
                    dim=dim, dim_head=dim_head, heads=heads,
                    dropout=attn_dropout, add_value_residual=add_value_residual,
                )
            else:
                attn = Attention(
                    dim=dim, dim_head=dim_head, heads=heads,
                    dropout=attn_dropout, rotary_embed=rotary_embed,
                    flash=flash_attn, add_value_residual=add_value_residual,
                )
            ff = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
            self.layers.append(nn.ModuleList([attn, ff]))

        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        value_residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        first_values: Optional[torch.Tensor] = None

        for attn, ff in self.layers:          # type: ignore[assignment]
            attn_out, values = attn(x, value_residual=value_residual)
            first_values = values if first_values is None else first_values
            x = x + attn_out
            x = x + ff(x)

        return self.norm(x), first_values      # type: ignore[return-value]


# ---------------------------------------------------------------------------
# RoFormer Backbone (bucle axial: linear + time + freq)
# ---------------------------------------------------------------------------

class RoFormerBackbone(nn.Module):
    """Backbone de atencion axial de Mel-RoFormer.

    Recibe features de banda (B, T, F, D) y aplica por cada capa:
      1. Linear Transformer  – atiende sobre T*F aplanado  (global, O(n))
      2. Time  Transformer   – atiende sobre T por cada F  (con RoPE)
      3. Freq  Transformer   – atiende sobre F por cada T  (con RoPE)

    RoPE es independiente para tiempo y frecuencia (no compartido).
    """

    def __init__(self, config: FeatureConfig) -> None:
        super().__init__()

        time_rotary = RotaryEmbedding(dim=config.dim_head)
        freq_rotary = RotaryEmbedding(dim=config.dim_head)

        tf_kwargs = dict(
            dim=config.dim,
            heads=config.heads,
            dim_head=config.dim_head,
            attn_dropout=config.attn_dropout,
            ff_dropout=config.ff_dropout,
            ff_mult=config.ff_mult,
            flash_attn=config.flash_attn,
        )

        self.layers = nn.ModuleList()
        for i in range(config.depth):
            # La primera capa NO usa value residual (no hay capa anterior)
            avr = config.add_value_residual and (i > 0)

            lin_tf: Optional[nn.Module] = (
                Transformer(
                    depth=config.linear_transformer_depth,
                    linear_attn=True,
                    add_value_residual=avr,
                    **tf_kwargs,
                )
                if config.linear_transformer_depth > 0
                else None
            )
            time_tf = Transformer(
                depth=config.time_transformer_depth,
                rotary_embed=time_rotary,
                add_value_residual=avr,
                **tf_kwargs,
            )
            freq_tf = Transformer(
                depth=config.freq_transformer_depth,
                rotary_embed=freq_rotary,
                add_value_residual=avr,
                **tf_kwargs,
            )
            self.layers.append(nn.ModuleList([lin_tf, time_tf, freq_tf]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, F, D)
        Returns:
            (B, T, F, D)
        """
        linear_vr: Optional[torch.Tensor] = None
        time_vr: Optional[torch.Tensor] = None
        freq_vr: Optional[torch.Tensor] = None

        for lin_tf, time_tf, freq_tf in self.layers:  # type: ignore[assignment]
            b, t, f, d = x.shape

            # --- Linear Transformer (global sobre T*F aplanado) ---
            if lin_tf is not None:
                x_flat = x.reshape(b, t * f, d)
                x_flat, next_lv = lin_tf(x_flat, value_residual=linear_vr)
                linear_vr = linear_vr if linear_vr is not None else next_lv
                x = x_flat.reshape(b, t, f, d)

            # --- Time Transformer (atiende sobre T por cada F) ---
            x_t = rearrange(x, "b t f d -> (b f) t d")
            x_t, next_tv = time_tf(x_t, value_residual=time_vr)
            time_vr = time_vr if time_vr is not None else next_tv
            x = rearrange(x_t, "(b f) t d -> b t f d", b=b, f=f)

            # --- Freq Transformer (atiende sobre F por cada T) ---
            x_f = rearrange(x, "b t f d -> (b t) f d")
            x_f, next_fv = freq_tf(x_f, value_residual=freq_vr)
            freq_vr = freq_vr if freq_vr is not None else next_fv
            x = rearrange(x_f, "(b t) f d -> b t f d", b=b, t=t)

        return x


# ---------------------------------------------------------------------------
# MLP helper
# ---------------------------------------------------------------------------

def _mlp(
    dim_in: int,
    dim_out: int,
    dim_hidden: Optional[int] = None,
    depth: int = 1,
) -> nn.Sequential:
    dim_hidden = dim_hidden or dim_in
    dims = [dim_in] + [dim_hidden] * depth + [dim_out]
    layers: list[nn.Module] = []
    for i, (in_d, out_d) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(in_d, out_d))
        if i < len(dims) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Mask Estimator con MLP + GLU (unico stem)
# ---------------------------------------------------------------------------

class MaskEstimator(nn.Module):
    """Estimador de mascara compleja por banda mel (MLP + GLU).

    Equivalente al MaskEstimator de BS-RoFormer pero para un unico stem.
    Cada banda tiene su propia MLP independiente para mayor capacidad.
    """

    def __init__(
        self,
        dim: int,
        dim_inputs: Tuple[int, ...],
        depth: int = 2,
        mlp_expansion_factor: int = 4,
    ) -> None:
        super().__init__()
        dim_hidden = dim * mlp_expansion_factor
        self.to_freqs = nn.ModuleList(
            [
                nn.Sequential(
                    _mlp(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth),
                    nn.GLU(dim=-1),
                )
                for dim_in in dim_inputs
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, F, D)  features post-backbone
        Returns:
            (B, T, sum(dim_inputs))  mascara plana real+imag
        """
        bands = x.unbind(dim=-2)     # F tensores de (B, T, D)
        outs = [mlp(band) for mlp, band in zip(self.to_freqs, bands)]
        return torch.cat(outs, dim=-1)
