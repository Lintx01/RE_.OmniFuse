from __future__ import annotations

import argparse
from typing import Dict

import torch

from .model import OmniFuse


def make_synth_batch(
    batch_size: int,
    input_dims: Dict[str, int],
    num_classes: int,
    device: torch.device,
    missing_prob: float,
):
    x: Dict[str, torch.Tensor] = {m: torch.randn(batch_size, d, device=device) for m, d in input_dims.items()}

    # Create labels from a latent ground-truth linear rule to make the task learnable.
    # (Only for synthetic sanity-check.)
    y_logits = torch.zeros(batch_size, num_classes, device=device)
    for i, (m, d) in enumerate(input_dims.items()):
        W = torch.randn(d, num_classes, device=device) * (0.3 + 0.1 * i)
        y_logits = y_logits + x[m] @ W
    y = torch.argmax(y_logits, dim=-1)

    # Random missingness per modality (ensure at least 1 modality available)
    available: Dict[str, torch.Tensor] = {}
    for m in input_dims.keys():
        available[m] = (torch.rand(batch_size, device=device) > missing_prob)

    all_missing = torch.ones(batch_size, device=device, dtype=torch.bool)
    for m in input_dims.keys():
        all_missing = all_missing & (~available[m])
    if torch.any(all_missing):
        first_m = next(iter(input_dims.keys()))
        available[first_m][all_missing] = True

    return x, available, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--missing-prob", type=float, default=0.4)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use 3 modalities to match the paper's common setting (IMG/EHR/REP), but keep dims small.
    input_dims = {"img": 128, "ehr": 64, "rep": 160}
    embed_dims = {"img": 64, "ehr": 64, "rep": 64}

    model = OmniFuse(
        input_dims=input_dims,
        embed_dims=embed_dims,
        num_classes=args.num_classes,
        num_ras=5,
        ra_latent_dim=64,
        ra_hidden_dim=256,
        alpha=0.1,
        beta=0.5,
        lambda1=10.0,
        lambda2=1.0,
        lambda3=0.5,
        dropout=0.0,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for step in range(1, args.steps + 1):
        x, available, y = make_synth_batch(
            batch_size=args.batch_size,
            input_dims=input_dims,
            num_classes=args.num_classes,
            device=device,
            missing_prob=args.missing_prob,
        )

        opt.zero_grad(set_to_none=True)
        logits, losses = model(x=x, available=available, y=y, compute_losses=True)
        losses.total.backward()
        opt.step()

        if step % 20 == 0 or step == 1:
            with torch.no_grad():
                pred = torch.argmax(logits, dim=-1)
                acc = (pred == y).float().mean().item()
            print(
                f"step={step:04d} "
                f"loss={losses.total.item():.4f} "
                f"L_fwd={losses.l_forward.item():.4f} "
                f"L_bwd={losses.l_backward.item():.4f} "
                f"L_enc={losses.l_enc.item():.4f} "
                f"L_ra={losses.l_ra.item():.4f} "
                f"L_tla={losses.l_tla.item():.4f} "
                f"acc={acc:.3f}"
            )


if __name__ == "__main__":
    main()
