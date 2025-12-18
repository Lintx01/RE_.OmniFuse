from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


class TabularMLPEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResNetImageEncoder(nn.Module):
    def __init__(self, out_dim: int, backbone: str = "resnet34", pretrained: bool = True, dropout: float = 0.0):
        super().__init__()

        try:
            import torchvision.models as models
        except Exception as e:  # pragma: no cover
            raise RuntimeError("需要安装 torchvision 才能使用 ResNetImageEncoder") from e

        if backbone == "resnet18":
            net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        elif backbone == "resnet34":
            net = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        elif backbone == "resnet50":
            net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        else:
            raise ValueError(f"不支持的 backbone: {backbone}")

        in_feat = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net
        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feat, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.proj(feats)


class MultiViewImageEncoder(nn.Module):
    """Encode multiple ultrasound images per patient and aggregate.

    Input:
    - images: FloatTensor [B, K, 3, H, W]
    - mask: BoolTensor [B, K] (True = valid)

    Output:
    - embedding: FloatTensor [B, out_dim]
    """

    def __init__(self, out_dim: int, backbone: str = "resnet34", pretrained: bool = True, dropout: float = 0.0):
        super().__init__()
        self.single = ResNetImageEncoder(out_dim=out_dim, backbone=backbone, pretrained=pretrained, dropout=dropout)

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5:
            raise ValueError(f"期望 images 为 [B,K,3,H,W]，实际 {images.shape}")
        if mask.ndim != 2:
            raise ValueError(f"期望 mask 为 [B,K]，实际 {mask.shape}")

        B, K = images.shape[0], images.shape[1]
        flat = images.view(B * K, *images.shape[2:])
        emb = self.single(flat).view(B, K, -1)

        m = mask.to(dtype=emb.dtype).unsqueeze(-1)  # [B,K,1]
        denom = m.sum(dim=1).clamp(min=1.0)
        pooled = (emb * m).sum(dim=1) / denom
        return pooled


@dataclass(frozen=True)
class TextEncoderOutput:
    embedding: torch.Tensor


class HFTextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "bert-base-chinese",
        out_dim: int = 128,
        max_length: int = 256,
        dropout: float = 0.0,
        trainable: bool = True,
    ):
        super().__init__()

        try:
            import importlib

            transformers = importlib.import_module("transformers")
            auto_model = getattr(transformers, "AutoModel")
            auto_tokenizer = getattr(transformers, "AutoTokenizer")
        except Exception as e:  # pragma: no cover
            raise RuntimeError("需要安装 transformers 才能使用 HFTextEncoder") from e

        self.tokenizer = auto_tokenizer.from_pretrained(model_name, use_fast=True)
        self.model = auto_model.from_pretrained(model_name)
        self.max_length = max_length

        if not trainable:
            for p in self.model.parameters():
                p.requires_grad = False

        hidden = self.model.config.hidden_size
        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, text_list: list[str]) -> TextEncoderOutput:
        device = next(self.parameters()).device
        batch = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        out = self.model(**batch)

        # Prefer pooler_output if exists, else CLS
        pooled = getattr(out, "pooler_output", None)
        if pooled is None:
            pooled = out.last_hidden_state[:, 0]

        emb = self.proj(pooled)
        return TextEncoderOutput(embedding=emb)
