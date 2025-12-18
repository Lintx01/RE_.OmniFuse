from __future__ import annotations

import argparse

import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from .datasets_raw import RawTriModalManifestDataset, collate_raw_trifuse
from .encoders import HFTextEncoder, MultiViewImageEncoder, TabularMLPEncoder
from .model_raw import RawOmniFuse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True, help="CSV 列：id,label,labs,us,text")
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-us-images", type=int, default=8, help="每个病人最多读取多少张超声图像（多角度）")
    parser.add_argument("--image-backbone", type=str, default="resnet34", choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--no-pretrained-image", action="store_true")
    parser.add_argument("--labs-table", type=str, default=None, help="实验室总表 CSV（包含 id 列 + 多个数值列）")
    parser.add_argument("--labs-id-col", type=str, default="id")
    parser.add_argument("--text-model", type=str, default="bert-base-chinese")
    parser.add_argument("--text-max-length", type=int, default=256)
    parser.add_argument("--freeze-text", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = RawTriModalManifestDataset(
        args.manifest,
        image_size=args.image_size,
        max_us_images=args.max_us_images,
        labs_table_csv=args.labs_table,
        labs_id_col=args.labs_id_col,
    )

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_raw_trifuse,
        pin_memory=(device.type == "cuda"),
    )

    labs_encoder = TabularMLPEncoder(in_dim=ds.labs_dim, out_dim=args.embed_dim, hidden_dim=256, dropout=0.1).to(device)
    us_encoder = MultiViewImageEncoder(
        out_dim=args.embed_dim,
        backbone=args.image_backbone,
        pretrained=(not args.no_pretrained_image),
        dropout=0.0,
    ).to(device)
    text_encoder = HFTextEncoder(
        model_name=args.text_model,
        out_dim=args.embed_dim,
        max_length=args.text_max_length,
        dropout=0.0,
        trainable=(not args.freeze_text),
    ).to(device)

    model = RawOmniFuse(
        labs_encoder=labs_encoder,
        us_encoder=us_encoder,
        text_encoder=text_encoder,
        embed_dim=args.embed_dim,
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

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        all_pred = []
        all_y = []

        for x, available, y in dl:
            x["labs"] = x["labs"].to(device)
            x["us"] = x["us"].to(device)
            x["us_mask"] = x["us_mask"].to(device)
            # x['text'] is list[str]
            available = {k: v.to(device) for k, v in available.items()}
            y = y.to(device)

            opt.zero_grad(set_to_none=True)
            logits, l = model(x=x, available=available, y=y, compute_losses=True)
            l.total.backward()
            opt.step()

            losses.append(float(l.total.item()))
            pred = torch.argmax(logits, dim=-1)
            all_pred.extend(pred.detach().cpu().tolist())
            all_y.extend(y.detach().cpu().tolist())

        acc = accuracy_score(all_y, all_pred) if len(all_y) else 0.0
        print(f"epoch={epoch:03d} loss={sum(losses)/max(len(losses),1):.4f} acc={acc:.3f}")


if __name__ == "__main__":
    main()
