from __future__ import annotations

import argparse
import csv
import os

import torch
import numpy as np
from torch.utils.data import Subset
from torch.utils.data import DataLoader

from .datasets_raw import RawTriModalManifestDataset, collate_raw_trifuse
from .encoders import HFTextEncoder, MultiViewImageEncoder, SimpleHashTextEncoder, TabularMLPEncoder
from .model_raw import RawOmniFuse


def _stratified_split_indices(labels: list[int], train: float, val: float, test: float, seed: int):
    if abs((train + val + test) - 1.0) > 1e-6:
        raise ValueError("train+val+test 必须等于 1")

    rng = np.random.default_rng(seed)
    labels_arr = np.asarray(labels, dtype=np.int64)

    train_idx = []
    val_idx = []
    test_idx = []

    for lab in np.unique(labels_arr):
        idx = np.where(labels_arr == lab)[0]
        rng.shuffle(idx)
        n = idx.shape[0]
        n_train = int(round(n * train))
        n_val = int(round(n * val))
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        n_test = n - n_train - n_val

        train_idx.extend(idx[:n_train].tolist())
        val_idx.extend(idx[n_train : n_train + n_val].tolist())
        test_idx.extend(idx[n_train + n_val :].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return train_idx, val_idx, test_idx


def _confusion_matrix(y_true: list[int], y_pred: list[int], num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def _macro_f1_from_cm(cm: np.ndarray) -> float:
    num_classes = cm.shape[0]
    f1s = []
    for c in range(num_classes):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - cm[c, c])
        fn = float(cm[c, :].sum() - cm[c, c])
        denom = (2 * tp + fp + fn)
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


@torch.no_grad()
def _eval(model: RawOmniFuse, loader: DataLoader, device: torch.device, num_classes: int):
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    losses = []
    for x, available, y in loader:
        x["labs"] = x["labs"].to(device)
        x["us"] = x["us"].to(device)
        x["us_mask"] = x["us_mask"].to(device)
        available = {k: v.to(device) for k, v in available.items()}
        y = y.to(device)

        logits, l = model(x=x, available=available, y=y, compute_losses=True)
        losses.append(float(l.total.item()))
        pred = torch.argmax(logits, dim=-1)
        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend(pred.detach().cpu().tolist())

    acc = float(np.mean([1.0 if a == b else 0.0 for a, b in zip(y_true, y_pred)])) if y_true else 0.0
    cm = _confusion_matrix(y_true, y_pred, num_classes=num_classes)
    macro_f1 = _macro_f1_from_cm(cm)
    avg_loss = float(np.mean(losses)) if losses else 0.0
    return avg_loss, acc, macro_f1, cm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True, help="CSV 列：id,label,labs,us,text")
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-image", type=float, default=1e-5, help="图像 backbone 的学习率（微调阶段）")
    parser.add_argument("--lr-text", type=float, default=1e-5, help="文本 backbone 的学习率（微调阶段）")
    parser.add_argument("--warmup-epochs", type=int, default=10, help="前 N 个 epoch 冻结 image/text backbone")
    parser.add_argument(
        "--finetune-image-scope",
        type=str,
        default="layer4",
        choices=["layer4", "all"],
        help="微调阶段图像 backbone 解冻范围：layer4=仅最后 stage，all=全解冻",
    )
    parser.add_argument(
        "--finetune-text-last-n",
        type=int,
        default=4,
        help="微调阶段文本模型解冻最后 N 层 Transformer（0=不解冻）",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="若提供，则把每个 epoch 的指标、最终 test 指标、混淆矩阵写入该目录（CSV）",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-us-images", type=int, default=8, help="每个病人最多读取多少张超声图像（多角度）")
    parser.add_argument("--image-backbone", type=str, default="resnet34", choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--no-pretrained-image", action="store_true")
    parser.add_argument("--labs-table", type=str, default=None, help="实验室总表 CSV（包含 id 列 + 多个数值列）")
    parser.add_argument("--labs-id-col", type=str, default="id")
    parser.add_argument("--text-model", type=str, default="bert-base-chinese")
    parser.add_argument("--text-max-length", type=int, default=256)
    parser.add_argument("--freeze-text", action="store_true")
    parser.add_argument("--text-encoder", type=str, default="simple", choices=["simple", "hf"], help="simple=离线可跑，hf=BERT")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    def _set_requires_grad(module: torch.nn.Module, flag: bool) -> None:
        for p in module.parameters():
            p.requires_grad = flag

    def _configure_image_backbone_trainable(us_enc: MultiViewImageEncoder, scope: str) -> None:
        # Always keep the projection head trainable
        _set_requires_grad(us_enc.single.proj, True)
        # Start frozen
        _set_requires_grad(us_enc.single.backbone, False)
        if scope == "all":
            _set_requires_grad(us_enc.single.backbone, True)
        elif scope == "layer4":
            layer4 = getattr(us_enc.single.backbone, "layer4", None)
            if layer4 is None:
                # fallback: unfreeze all if structure unexpected
                _set_requires_grad(us_enc.single.backbone, True)
            else:
                _set_requires_grad(layer4, True)
        else:
            raise ValueError(f"unknown image scope: {scope}")

    def _get_transformer_layers(hf_model: torch.nn.Module):
        # Works for BERT/RoBERTa style models: model.encoder.layer
        enc = getattr(hf_model, "encoder", None)
        if enc is not None and hasattr(enc, "layer"):
            return enc.layer
        # Some models: model.transformer.layer
        tr = getattr(hf_model, "transformer", None)
        if tr is not None and hasattr(tr, "layer"):
            return tr.layer
        return None

    def _configure_text_backbone_trainable(text_enc: torch.nn.Module, last_n: int) -> None:
        # Always keep projection trainable if present
        proj = getattr(text_enc, "proj", None)
        if isinstance(proj, torch.nn.Module):
            _set_requires_grad(proj, True)

        base = getattr(text_enc, "model", None)
        if not isinstance(base, torch.nn.Module):
            return

        _set_requires_grad(base, False)
        if last_n <= 0:
            return

        layers = _get_transformer_layers(base)
        if layers is None:
            # fallback: if we can't locate layers, unfreeze all
            _set_requires_grad(base, True)
            return

        n = min(int(last_n), len(layers))
        for layer in list(layers)[-n:]:
            _set_requires_grad(layer, True)

    def _build_optimizer(model_: RawOmniFuse, lr_main: float, lr_img: float, lr_txt: float) -> torch.optim.Optimizer:
        main_params = []
        img_params = []
        txt_params = []

        # Main = everything except trainable backbone params
        img_backbone = model_.us_encoder.single.backbone
        txt_backbone = getattr(model_.text_encoder, "model", None)

        img_backbone_param_ids = {id(p) for p in img_backbone.parameters()}
        txt_backbone_param_ids = {id(p) for p in txt_backbone.parameters()} if isinstance(txt_backbone, torch.nn.Module) else set()

        for p in model_.parameters():
            if not p.requires_grad:
                continue
            pid = id(p)
            if pid in img_backbone_param_ids:
                img_params.append(p)
            elif pid in txt_backbone_param_ids:
                txt_params.append(p)
            else:
                main_params.append(p)

        param_groups = []
        if main_params:
            param_groups.append({"params": main_params, "lr": lr_main})
        if img_params:
            param_groups.append({"params": img_params, "lr": lr_img})
        if txt_params:
            param_groups.append({"params": txt_params, "lr": lr_txt})
        return torch.optim.AdamW(param_groups)

    out_dir = None
    if args.out_dir:
        out_dir = os.path.abspath(args.out_dir)
        os.makedirs(out_dir, exist_ok=True)

        meta_path = os.path.join(out_dir, "meta.csv")
        with open(meta_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["key", "value"])
            w.writerow(["manifest", args.manifest])
            w.writerow(["num_classes", args.num_classes])
            w.writerow(["embed_dim", args.embed_dim])
            w.writerow(["epochs", args.epochs])
            w.writerow(["batch_size", args.batch_size])
            w.writerow(["lr", args.lr])
            w.writerow(["lr_image", args.lr_image])
            w.writerow(["lr_text", args.lr_text])
            w.writerow(["warmup_epochs", args.warmup_epochs])
            w.writerow(["finetune_image_scope", args.finetune_image_scope])
            w.writerow(["finetune_text_last_n", args.finetune_text_last_n])
            w.writerow(["seed", args.seed])
            w.writerow(["train_ratio", args.train_ratio])
            w.writerow(["val_ratio", args.val_ratio])
            w.writerow(["test_ratio", args.test_ratio])
            w.writerow(["image_size", args.image_size])
            w.writerow(["max_us_images", args.max_us_images])
            w.writerow(["image_backbone", args.image_backbone])
            w.writerow(["pretrained_image", (not args.no_pretrained_image)])
            w.writerow(["labs_table", args.labs_table or ""]) 
            w.writerow(["labs_id_col", args.labs_id_col])
            w.writerow(["text_encoder", args.text_encoder])
            w.writerow(["text_model", args.text_model])
            w.writerow(["text_max_length", args.text_max_length])
            w.writerow(["freeze_text", args.freeze_text])
            w.writerow(["num_workers", args.num_workers])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = RawTriModalManifestDataset(
        args.manifest,
        image_size=args.image_size,
        max_us_images=args.max_us_images,
        labs_table_csv=args.labs_table,
        labs_id_col=args.labs_id_col,
    )

    labels = [r.label for r in ds.rows]
    train_idx, val_idx, test_idx = _stratified_split_indices(
        labels, train=args.train_ratio, val=args.val_ratio, test=args.test_ratio, seed=args.seed
    )

    if out_dir is not None:
        split_path = os.path.join(out_dir, "split.csv")
        with open(split_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["split", "n"])
            w.writerow(["train", len(train_idx)])
            w.writerow(["val", len(val_idx)])
            w.writerow(["test", len(test_idx)])

    train_dl = DataLoader(
        Subset(ds, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_raw_trifuse,
        pin_memory=(device.type == "cuda"),
    )
    val_dl = DataLoader(
        Subset(ds, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_raw_trifuse,
        pin_memory=(device.type == "cuda"),
    )
    test_dl = DataLoader(
        Subset(ds, test_idx),
        batch_size=args.batch_size,
        shuffle=False,
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
    if args.text_encoder == "hf":
        text_encoder = HFTextEncoder(
            model_name=args.text_model,
            out_dim=args.embed_dim,
            max_length=args.text_max_length,
            dropout=0.0,
            trainable=True,
        ).to(device)
    else:
        text_encoder = SimpleHashTextEncoder(out_dim=args.embed_dim, vocab_size=5000, dropout=0.0).to(device)

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

    # Stage configuration
    # Warmup: freeze image/text backbone. Finetune: unfreeze per args.
    _set_requires_grad(model.us_encoder.single.backbone, False)
    if args.text_encoder == "hf":
        _set_requires_grad(model.text_encoder.model, False)
    # projection heads should remain trainable
    _set_requires_grad(model.us_encoder.single.proj, True)
    if hasattr(model.text_encoder, "proj"):
        _set_requires_grad(model.text_encoder.proj, True)

    opt = _build_optimizer(model, lr_main=args.lr, lr_img=args.lr_image, lr_txt=args.lr_text)
    stage = "warmup"

    metrics_path = None
    if out_dir is not None:
        metrics_path = os.path.join(out_dir, "metrics.csv")
        with open(metrics_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1"])

    for epoch in range(1, args.epochs + 1):
        # Switch to finetune stage at boundary
        if stage == "warmup" and epoch == (args.warmup_epochs + 1):
            stage = "finetune"
            _configure_image_backbone_trainable(model.us_encoder, scope=args.finetune_image_scope)
            if args.text_encoder == "hf":
                if args.freeze_text:
                    _set_requires_grad(model.text_encoder.model, False)
                else:
                    _configure_text_backbone_trainable(model.text_encoder, last_n=args.finetune_text_last_n)
            opt = _build_optimizer(model, lr_main=args.lr, lr_img=args.lr_image, lr_txt=args.lr_text)
            print(
                f"[stage=finetune] image={args.finetune_image_scope} text_last_n={0 if args.freeze_text else args.finetune_text_last_n} "
                f"lr={args.lr} lr_image={args.lr_image} lr_text={args.lr_text}"
            )

        model.train()
        losses = []

        for x, available, y in train_dl:
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

        train_loss = float(np.mean(losses)) if losses else 0.0
        val_loss, val_acc, val_f1, _ = _eval(model, val_dl, device=device, num_classes=args.num_classes)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_macro_f1={val_f1:.3f} "
            f"split={len(train_idx)}/{len(val_idx)}/{len(test_idx)}"
        )

        if metrics_path is not None:
            with open(metrics_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{val_acc:.6f}", f"{val_f1:.6f}"])

    test_loss, test_acc, test_f1, cm = _eval(model, test_dl, device=device, num_classes=args.num_classes)
    print(f"test_loss={test_loss:.4f} test_acc={test_acc:.3f} test_macro_f1={test_f1:.3f}")
    print("confusion_matrix (rows=true, cols=pred):")
    print(cm)

    if out_dir is not None:
        test_path = os.path.join(out_dir, "test.csv")
        with open(test_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["test_loss", "test_acc", "test_macro_f1"])
            w.writerow([f"{test_loss:.6f}", f"{test_acc:.6f}", f"{test_f1:.6f}"])

        cm_path = os.path.join(out_dir, "confusion_matrix.csv")
        with open(cm_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["true\\pred"] + [str(i) for i in range(args.num_classes)])
            for i in range(args.num_classes):
                w.writerow([str(i)] + [str(int(x)) for x in cm[i].tolist()])

        print(f"saved_results_dir={out_dir}")


if __name__ == "__main__":
    main()
