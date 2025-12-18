from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image


def _rand_cn_text(rng: random.Random, min_len: int = 30, max_len: int = 120) -> str:
    # Very small synthetic Chinese-like text (not medical).
    # Keep it simple and deterministic.
    vocab = list("患者复查随访正常轻症重症尿检血检超声提示考虑建议治疗观察无明显异常")
    L = rng.randint(min_len, max_len)
    return "".join(rng.choice(vocab) for _ in range(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default="data_synth_raw")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--labs-dim", type=int, default=40, help="实验室总表特征维度")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--max-us-images", type=int, default=5)

    ap.add_argument("--p-miss-us", type=float, default=0.2)
    ap.add_argument("--p-miss-text", type=float, default=0.1)
    ap.add_argument("--p-miss-labs", type=float, default=0.0, help="总表里某些患者 labs 缺失比例")

    args = ap.parse_args()

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    out_dir = Path(args.out_dir)
    us_dir = out_dir / "us"
    text_dir = out_dir / "text"
    out_dir.mkdir(parents=True, exist_ok=True)
    us_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.csv"
    labs_table_path = out_dir / "labs_table.csv"

    # labs_table.csv: id + f0..f{D-1}
    feature_cols = [f"f{i}" for i in range(args.labs_dim)]
    with labs_table_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", *feature_cols])
        w.writeheader()
        for i in range(args.n):
            pid = str(i + 1).zfill(10)
            if rng.random() < args.p_miss_labs:
                # skip this patient from total table to simulate missing
                continue
            vec = np_rng.normal(size=(args.labs_dim,)).astype(np.float32)
            row = {"id": pid}
            for j, c in enumerate(feature_cols):
                row[c] = f"{float(vec[j]):.6f}"
            w.writerow(row)

    # manifest.csv: id,label,labs,us,text
    # - labs 留空，让 Dataset 从 labs_table.csv 取
    # - us 多张图用 ';' 分隔
    # - text 写 txt 路径
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "label", "labs", "us", "text"])
        w.writeheader()

        # Ensure at least one sample has us+text present
        force_full_idx = 0

        for i in range(args.n):
            pid = str(i + 1).zfill(10)
            label = rng.randint(0, 2)

            miss_us = rng.random() < args.p_miss_us
            miss_text = rng.random() < args.p_miss_text
            if i == force_full_idx:
                miss_us = False
                miss_text = False

            us_cell = ""
            if not miss_us:
                k = rng.randint(1, args.max_us_images)
                paths = []
                for vi in range(k):
                    # random RGB image
                    arr = (np_rng.random((args.image_size, args.image_size, 3)) * 255).astype(np.uint8)
                    img = Image.fromarray(arr).convert("RGB")
                    rel = f"us/{pid}_{vi}.png"
                    img.save(out_dir / rel)
                    paths.append(rel)
                us_cell = ";".join(paths)

            text_cell = ""
            if not miss_text:
                rel = f"text/{pid}.txt"
                (out_dir / rel).write_text(_rand_cn_text(rng), encoding="utf-8")
                text_cell = rel

            w.writerow(
                {
                    "id": pid,
                    "label": str(label),
                    "labs": "",
                    "us": us_cell,
                    "text": text_cell,
                }
            )

    print(f"wrote: {manifest_path.resolve()}")
    print(f"wrote: {labs_table_path.resolve()}")
    print(f"n={args.n} labs_dim={args.labs_dim} image_size={args.image_size} max_us_images={args.max_us_images}")


if __name__ == "__main__":
    main()
