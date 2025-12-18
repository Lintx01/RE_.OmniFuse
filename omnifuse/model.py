from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import (
    CascadeResidualAutoencoder,
    DynamicWeightedFuse,
    MLPEncoder,
    all_non_empty_subsets,
    bitmask_from_availability,
)


@dataclass
class OmniFuseLosses:
    total: torch.Tensor
    l_forward: torch.Tensor
    l_backward: torch.Tensor
    l_enc: torch.Tensor
    l_ra: torch.Tensor
    l_tla: torch.Tensor


class OmniFuse(nn.Module):
    """A runnable implementation skeleton of OmniFuse.

    Components (per paper):
    - Modality-specific encoders (Eq. 1)
    - Missing modality imputation via CRA with forward/backward + cycle consistency (Eq. 2-8)
    - Dynamic Weighted Fuse (Eq. 9-14)
    - Traceable Laziness Activation (Eq. 15-17)
    - Total objective (Eq. 18)

    Notes:
    - This implementation focuses on matching the *training mechanics* and module interfaces.
    - For real-world tasks, replace MLP encoders with ResNet/BERT/etc.
    """

    def __init__(
        self,
        input_dims: Dict[str, int],
        embed_dims: Dict[str, int],
        num_classes: int,
        num_ras: int = 5,
        ra_latent_dim: int = 128,
        ra_hidden_dim: int = 512,
        alpha: float = 0.1,
        beta: float = 0.5,
        lambda1: float = 10.0,
        lambda2: float = 1.0,
        lambda3: float = 0.5,
        dropout: float = 0.0,
    ):
        super().__init__()

        if set(input_dims.keys()) != set(embed_dims.keys()):
            raise ValueError("input_dims 和 embed_dims 的模态集合必须一致")

        self.modalities: List[str] = list(input_dims.keys())
        self.num_classes = num_classes

        self.alpha = alpha
        self.beta = beta
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        self.encoders = nn.ModuleDict(
            {m: MLPEncoder(input_dims[m], embed_dims[m], hidden_dim=max(256, embed_dims[m]), dropout=dropout) for m in self.modalities}
        )

        self.concat_dim = sum(embed_dims[m] for m in self.modalities)
        self.embed_dims = embed_dims

        self.forward_cra = CascadeResidualAutoencoder(
            in_dim=self.concat_dim,
            num_ras=num_ras,
            latent_dim=ra_latent_dim,
            hidden_dim=ra_hidden_dim,
            dropout=dropout,
        )
        self.backward_cra = CascadeResidualAutoencoder(
            in_dim=self.concat_dim,
            num_ras=num_ras,
            latent_dim=ra_latent_dim,
            hidden_dim=ra_hidden_dim,
            dropout=dropout,
        )

        self.dw_fuse = DynamicWeightedFuse(embed_dims=embed_dims, num_classes=num_classes, beta=beta)

        self.ra_head = nn.Linear(num_ras * ra_latent_dim, num_classes)

    def _encode_modalities(
        self, x: Dict[str, torch.Tensor], available: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        embeds: Dict[str, torch.Tensor] = {}
        for m in self.modalities:
            v = self.encoders[m](x[m])
            # If missing, replace with zero vector embedding (paper: encode zero vectors)
            v = torch.where(available[m].unsqueeze(1), v, torch.zeros_like(v))
            embeds[m] = v
        return embeds

    def _concat_embeds(self, embeds: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([embeds[m] for m in self.modalities], dim=1)

    def _split_concat(self, v_concat: torch.Tensor) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        start = 0
        for m in self.modalities:
            d = self.embed_dims[m]
            out[m] = v_concat[:, start : start + d]
            start += d
        return out

    def forward(
        self,
        x: Dict[str, torch.Tensor],
        available: Dict[str, torch.Tensor],
        y: Optional[torch.Tensor] = None,
        compute_losses: bool = False,
    ) -> Tuple[torch.Tensor, Optional[OmniFuseLosses]]:
        """Return fused_logits and optional losses."""

        embeds_in = self._encode_modalities(x, available)
        v = self._concat_embeds(embeds_in)  # Eq (2): concat with missing zeros

        # Ground-truth full embeddings (Eq 3) — for synthetic, we can compute from raw x directly.
        # In real setting, this corresponds to embeddings extracted from modality-specific encoders with full availability.
        if compute_losses:
            if y is None:
                raise ValueError("compute_losses=True 时必须提供 y")
            embeds_full = {m: self.encoders[m](x[m]) for m in self.modalities}
            v_hat = self._concat_embeds(embeds_full)
        else:
            v_hat = None

        # Forward CRA (Eq 5) + collect RA latents for joint classifier (Eq 11)
        v_prime, ra_latents = self.forward_cra(v)
        embeds_imputed = self._split_concat(v_prime)

        # DWFuse (Eq 9-14): per modality logits + weighted fuse
        dw_out = self.dw_fuse(embeds_imputed, y=y if y is not None else torch.zeros(v.shape[0], device=v.device, dtype=torch.long))
        fused_logits = dw_out.fused_logits

        # RA joint classifier (Eq 11-13)
        ra_feat = torch.cat(ra_latents, dim=1)
        logits_ra = self.ra_head(ra_feat)

        if not compute_losses:
            return fused_logits, None

        # Backward CRA (Eq 6)
        v_double_prime, _ = self.backward_cra(v_prime)

        # Losses
        l_forward = F.mse_loss(v_prime, v_hat)
        l_backward = F.mse_loss(v_double_prime, v)

        l_enc = F.cross_entropy(fused_logits, y)
        l_ra = F.cross_entropy(logits_ra, y)
        l_dwfuse = l_enc + self.alpha * l_ra

        l_tla = self._tla_loss(x=x, y=y, available=available)

        total = l_forward + self.lambda1 * l_backward + self.lambda2 * l_dwfuse + self.lambda3 * l_tla

        return fused_logits, OmniFuseLosses(
            total=total,
            l_forward=l_forward,
            l_backward=l_backward,
            l_enc=l_enc,
            l_ra=l_ra,
            l_tla=l_tla,
        )

    @torch.no_grad()
    def _predict_with_subset(self, x: Dict[str, torch.Tensor], subset_bitmask: int, y: torch.Tensor) -> torch.Tensor:
        """Prediction logits when only a subset of modalities is available."""
        avail: Dict[str, torch.Tensor] = {}
        for i, m in enumerate(self.modalities):
            is_on = (subset_bitmask >> i) & 1
            avail[m] = torch.full((y.shape[0],), bool(is_on), device=y.device, dtype=torch.bool)
        logits, _ = self.forward(x=x, available=avail, y=y, compute_losses=False)
        return logits

    def _tla_loss(self, x: Dict[str, torch.Tensor], y: torch.Tensor, available: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Traceable Laziness Activation (TLA) loss, Eq (15)-(17).

        Practical implementation:
        - For all non-empty modality subsets, compute combination-wise distance via mean CE.
        - Pick the subset with the largest distance as δ_lazy.
        - Build mask selecting samples whose *current availability pattern* equals δ_lazy.
        - Apply extra CE on those masked samples.
        """

        subsets = all_non_empty_subsets(self.modalities)
        # combination-wise distances
        distances = []
        for bm in subsets:
            logits_bm = self._predict_with_subset(x=x, subset_bitmask=bm, y=y)
            loss_vec = F.cross_entropy(logits_bm, y, reduction="none")
            distances.append(loss_vec.mean())

        distances_t = torch.stack(distances)
        lazy_idx = int(torch.argmax(distances_t).item())
        delta_lazy = subsets[lazy_idx]

        pattern = bitmask_from_availability(self.modalities, available)
        mask = pattern == delta_lazy
        if not torch.any(mask):
            return torch.zeros((), device=y.device)

        fused_logits, _ = self.forward(x=x, available=available, y=y, compute_losses=False)
        return F.cross_entropy(fused_logits[mask], y[mask])
