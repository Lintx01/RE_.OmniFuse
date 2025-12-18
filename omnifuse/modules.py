from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualAutoencoder(nn.Module):
    """A single Residual Autoencoder (RA).

    In the paper's CRA, each RA produces a residual in the embedding space (Δz_k)
    and also yields a latent vector c_k.
    """

    def __init__(self, in_dim: int, latent_dim: int, hidden_dim: int = 512, dropout: float = 0.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        c = self.encoder(x)
        delta = self.decoder(c)
        return delta, c


class CascadeResidualAutoencoder(nn.Module):
    """Cascade Residual Autoencoder (CRA) with K Residual Autoencoders (RAs).

    Implements Eq. (4) and forward aggregation Eq. (5).
    """

    def __init__(self, in_dim: int, num_ras: int = 5, latent_dim: int = 128, hidden_dim: int = 512, dropout: float = 0.0):
        super().__init__()
        self.num_ras = num_ras
        self.ras = nn.ModuleList(
            [ResidualAutoencoder(in_dim, latent_dim, hidden_dim=hidden_dim, dropout=dropout) for _ in range(num_ras)]
        )

    def forward(self, v: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        deltas: List[torch.Tensor] = []
        latents: List[torch.Tensor] = []
        sum_delta = torch.zeros_like(v)
        for k, ra in enumerate(self.ras, start=1):
            inp = v if k == 1 else (v + sum_delta)
            delta, c = ra(inp)
            deltas.append(delta)
            latents.append(c)
            sum_delta = sum_delta + delta
        v_imputed = v + sum_delta
        return v_imputed, latents


@dataclass(frozen=True)
class DWFuseOutput:
    fused_logits: torch.Tensor
    modality_logits: Dict[str, torch.Tensor]
    omega: Dict[str, torch.Tensor]


class DynamicWeightedFuse(nn.Module):
    """Dynamic Weighted Fuse (DWFuse), Eq. (9)-(14).

    We compute per-modality logits, then use the other modalities' confidence
    (probability assigned to the correct class) to down-weight each modality.

    Practical note:
    - The paper defines ω_m using p_n (confidence of other modalities). Here we use
      p_n_correct = softmax(logits_n)[y] per-sample.
    - We apply ω_m to logits and sum them to get fused_logits.
    """

    def __init__(self, embed_dims: Dict[str, int], num_classes: int, beta: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.modalities = list(embed_dims.keys())
        self.num_classes = num_classes
        self.beta = beta
        self.eps = eps

        self.modality_heads = nn.ModuleDict({m: nn.Linear(embed_dims[m], num_classes) for m in self.modalities})

    def forward(self, embeds: Dict[str, torch.Tensor], y: torch.Tensor) -> DWFuseOutput:
        logits: Dict[str, torch.Tensor] = {m: self.modality_heads[m](embeds[m]) for m in self.modalities}

        # p_correct per modality
        p_correct: Dict[str, torch.Tensor] = {}
        for m in self.modalities:
            probs = F.softmax(logits[m], dim=-1)
            p_correct[m] = probs.gather(1, y.view(-1, 1)).squeeze(1)  # [B]

        # ω_m = [ prod_{n!=m} (1 - p_n) ]^(β/(M-1))
        M = len(self.modalities)
        omega: Dict[str, torch.Tensor] = {}
        for m in self.modalities:
            others = [n for n in self.modalities if n != m]
            if len(others) == 0:
                omega[m] = torch.ones_like(p_correct[m])
                continue
            log_term = 0.0
            for n in others:
                log_term = log_term + torch.log1p(-p_correct[n].clamp(min=0.0, max=1.0 - self.eps) + self.eps)
            omega_m = torch.exp((self.beta / (M - 1)) * log_term)
            omega[m] = omega_m

        fused_logits = torch.zeros_like(next(iter(logits.values())))
        for m in self.modalities:
            fused_logits = fused_logits + omega[m].unsqueeze(1) * logits[m]

        return DWFuseOutput(fused_logits=fused_logits, modality_logits=logits, omega=omega)


def all_non_empty_subsets(modalities: List[str]) -> List[int]:
    """Return bitmask list for all non-empty subsets."""
    masks = []
    M = len(modalities)
    for bitmask in range(1, 1 << M):
        masks.append(bitmask)
    return masks


def bitmask_from_availability(modalities: List[str], available: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Compute per-sample bitmask integer from availability dict.

    available[m] is BoolTensor [B].
    """
    B = next(iter(available.values())).shape[0]
    mask = torch.zeros(B, device=next(iter(available.values())).device, dtype=torch.long)
    for i, m in enumerate(modalities):
        mask = mask | ((available[m].long() & 1) << i)
    return mask
