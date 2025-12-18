from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import OmniFuse, OmniFuseLosses


@dataclass
class RawOmniFuseLosses(OmniFuseLosses):
    pass


class RawOmniFuse(nn.Module):
    """Wrap OmniFuse to support raw modalities:

    - labs: Float tensor [B, D]
    - us: Image tensor [B, 3, H, W]
    - text: List[str]

    Encoders are provided externally (tabular/image/text).
    """

    def __init__(
        self,
        labs_encoder: nn.Module,
        us_encoder: nn.Module,
        text_encoder: nn.Module,
        embed_dim: int,
        num_classes: int,
        **omnifuse_kwargs,
    ):
        super().__init__()
        self.labs_encoder = labs_encoder
        self.us_encoder = us_encoder
        self.text_encoder = text_encoder

        input_dims = {"labs": embed_dim, "us": embed_dim, "text": embed_dim}
        embed_dims = {"labs": embed_dim, "us": embed_dim, "text": embed_dim}

        self.core = OmniFuse(
            input_dims=input_dims,
            embed_dims=embed_dims,
            num_classes=num_classes,
            **omnifuse_kwargs,
        )

    def forward(
        self,
        x: Dict[str, object],
        available: Dict[str, torch.Tensor],
        y: Optional[torch.Tensor] = None,
        compute_losses: bool = False,
    ) -> Tuple[torch.Tensor, Optional[RawOmniFuseLosses]]:
        labs: torch.Tensor = x["labs"]  # [B, D]
        us: torch.Tensor = x["us"]  # [B, K, 3, H, W]
        us_mask: torch.Tensor = x.get("us_mask")  # [B, K]
        text_list: list[str] = x["text"]  # list[str]

        v_labs = self.labs_encoder(labs)
        if us_mask is None:
            raise ValueError("缺少 us_mask（多张超声图像聚合需要）")
        v_us = self.us_encoder(us, us_mask)
        v_text = self.text_encoder(text_list).embedding

        enc_x = {"labs": v_labs, "us": v_us, "text": v_text}

        logits, losses = self.core(enc_x, available=available, y=y, compute_losses=compute_losses)
        return logits, losses
