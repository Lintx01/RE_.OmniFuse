from __future__ import annotations

import argparse
from typing import Dict

import torch
from torch.utils.data import DataLoader

from .datasets import TriModalManifestDataset, collate_trifuse
from .model import OmniFuse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True, help="CSV 文件，列：id,label,labs,us,text")
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = TriModalManifestDataset(args.manifest)
    input_dims: Dict[str, int] = ds.input_dims
    embed_dims: Dict[str, int] = {m: args.embed_dim for m in input_dims.keys()}

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_trifuse,
        pin_memory=(device.type == "cuda"),
    )

    model = OmniFuse(
        input_dims=input_dims,
        embed_dims=embed_dims,
        num_classes=args.num_classes,
        num_ras=5,
        ra_latent_dim=min(64, args.embed_dim),
        ra_hidden_dim=max(256, args.embed_dim),
        alpha=0.1,
        beta=0.5,
        lambda1=10.0,
        lambda2=1.0,
        lambda3=0.5,
        dropout=0.1,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for x, available, y in dl:
            x = {k: v.to(device) for k, v in x.items()}
            available = {k: v.to(device) for k, v in available.items()}
            y = y.to(device)

            opt.zero_grad(set_to_none=True)
            _, losses = model(x=x, available=available, y=y, compute_losses=True)
            losses.total.backward()
            opt.step()

            running += float(losses.total.item()) * y.shape[0]
            n += y.shape[0]

        print(f"epoch={epoch:03d} loss={running / max(n, 1):.4f}")


if __name__ == "__main__":
    main()
