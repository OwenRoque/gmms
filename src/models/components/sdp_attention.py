"""Atencion con seleccion de kernel SDP (estilo BS-RoFormer attend.py)."""

from __future__ import annotations

from collections import namedtuple
from typing import Optional

import torch
import torch.nn.functional as F

FlashAttentionConfig = namedtuple(
    "FlashAttentionConfig",
    ["enable_flash", "enable_math", "enable_mem_efficient"],
)


def _sdp_config(device: torch.device) -> FlashAttentionConfig:
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        # A100 sm_80: flash exclusivo. Resto de GPUs: math + mem-efficient.
        if props.major == 8 and props.minor == 0:
            return FlashAttentionConfig(True, False, False)
        return FlashAttentionConfig(False, True, True)
    return FlashAttentionConfig(True, True, True)


def sdp_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    dropout_p: float = 0.0,
    flash: bool = True,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Wrapper sobre scaled_dot_product_attention con sdp_kernel configurable."""
    if not flash:
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
        )

    config = _sdp_config(q.device)
    if q.is_cuda and hasattr(torch.backends, "cuda"):
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
            )
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
    )
