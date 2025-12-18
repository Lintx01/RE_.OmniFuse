from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import hashlib

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

        # Mean pooling over tokens using attention_mask
        last = out.last_hidden_state  # [B, T, H]
        attn = batch.get("attention_mask")
        if attn is None:
            pooled = last.mean(dim=1)
        else:
            m = attn.to(dtype=last.dtype).unsqueeze(-1)  # [B, T, 1]
            denom = m.sum(dim=1).clamp(min=1.0)
            pooled = (last * m).sum(dim=1) / denom

        emb = self.proj(pooled)
        return TextEncoderOutput(embedding=emb)


class SimpleHashTextEncoder(nn.Module):
    """A tiny text encoder that doesn't require external models.

    It hashes tokens into a fixed vocab, embeds them, and averages per sample.
    Useful for quickly verifying the end-to-end pipeline without downloading BERT.
    """

    def __init__(self, out_dim: int = 128, vocab_size: int = 5000, dropout: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(vocab_size, out_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        t = (text or "").strip()
        if t == "":
            return []
        # simple whitespace split; if no spaces, fall back to character tokens
        toks = t.split()
        if len(toks) <= 1:
            toks = list(t)
        return toks

    def _hash_token(self, token: str) -> int:
        h = hashlib.md5(token.encode("utf-8")).hexdigest()
        return int(h, 16) % self.vocab_size

    def forward(self, text_list: list[str]) -> TextEncoderOutput:
        device = next(self.parameters()).device
        batch_embs = []
        for text in text_list:
            toks = self._tokenize(text)
            if len(toks) == 0:
                batch_embs.append(torch.zeros(self.emb.embedding_dim, device=device))
                continue
            idx = torch.tensor([self._hash_token(t) for t in toks], device=device, dtype=torch.long)
            e = self.emb(idx)  # [T, D]
            e = self.dropout(e)
            batch_embs.append(e.mean(dim=0))
        return TextEncoderOutput(embedding=torch.stack(batch_embs, dim=0))
