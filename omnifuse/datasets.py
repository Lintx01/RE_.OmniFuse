from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ManifestRow:
    sample_id: str
    label: int
    labs_path: Optional[str]
    us_path: Optional[str]
    text_path: Optional[str]


def normalize_id(value: str, width: int = 10) -> str:
    s = str(value).strip()
    if s.isdigit() and len(s) < width:
        return s.zfill(width)
    if s.isdigit() and len(s) > width:
        raise ValueError(f"ID 长度超过 {width} 位：{s}（请检查是否需要截断/清洗）")
    return s


def _to_optional_path(base_dir: str, value: str) -> Optional[str]:
    value = (value or "").strip()
    if value == "" or value.lower() in {"none", "null", "nan"}:
        return None
    path = value
    if not os.path.isabs(path):
        path = os.path.join(base_dir, path)
    return path


def _load_npy_vector(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 1:
        raise ValueError(f"期望 1D 向量特征，实际 {path} shape={arr.shape}")
    return arr.astype(np.float32)


class TriModalManifestDataset(Dataset):
    """三模态数据集：实验室向量/超声特征/文本特征。

    约定：使用一个 CSV manifest，至少包含这些列：
    - id: 样本唯一 ID
    - label: 分类标签（int）
    - labs: 实验室特征 .npy 路径（可空）
    - us: 超声特征 .npy 路径（可空）
    - text: 文本特征 .npy 路径（可空）

    缺失模态：对应列为空即可。
    """

    def __init__(self, manifest_csv: str):
        super().__init__()
        self.manifest_csv = manifest_csv
        self.base_dir = os.path.dirname(os.path.abspath(manifest_csv))

        rows: List[ManifestRow] = []
        with open(manifest_csv, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            required = {"id", "label", "labs", "us", "text"}
            if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
                raise ValueError(f"manifest 缺少列：需要 {sorted(required)}，实际 {reader.fieldnames}")
            for r in reader:
                rows.append(
                    ManifestRow(
                        sample_id=normalize_id(r["id"]),
                        label=int(r["label"]),
                        labs_path=_to_optional_path(self.base_dir, r.get("labs", "")),
                        us_path=_to_optional_path(self.base_dir, r.get("us", "")),
                        text_path=_to_optional_path(self.base_dir, r.get("text", "")),
                    )
                )
        if len(rows) == 0:
            raise ValueError("manifest 为空")
        self.rows = rows

        # Infer dims from first non-missing for each modality
        self.input_dims: Dict[str, int] = {}
        for key, attr in [("labs", "labs_path"), ("us", "us_path"), ("text", "text_path")]:
            dim = None
            for row in self.rows:
                p = getattr(row, attr)
                if p is None:
                    continue
                dim = int(_load_npy_vector(p).shape[0])
                break
            if dim is None:
                raise ValueError(f"模态 {key} 在整个数据集中都缺失，无法训练")
            self.input_dims[key] = dim

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
        row = self.rows[idx]

        x: Dict[str, torch.Tensor] = {}
        available: Dict[str, torch.Tensor] = {}

        for key, path in [("labs", row.labs_path), ("us", row.us_path), ("text", row.text_path)]:
            if path is None:
                vec = np.zeros((self.input_dims[key],), dtype=np.float32)
                avail = False
            else:
                vec = _load_npy_vector(path)
                if vec.shape[0] != self.input_dims[key]:
                    raise ValueError(f"维度不一致：{key} 期望 {self.input_dims[key]} 实际 {vec.shape[0]} ({path})")
                avail = True

            x[key] = torch.from_numpy(vec)
            available[key] = torch.tensor(avail, dtype=torch.bool)

        y = torch.tensor(row.label, dtype=torch.long)
        return x, available, y


def collate_trifuse(batch):
    xs, avs, ys = zip(*batch)

    out_x: Dict[str, torch.Tensor] = {}
    out_av: Dict[str, torch.Tensor] = {}

    modalities = xs[0].keys()
    for m in modalities:
        out_x[m] = torch.stack([x[m] for x in xs], dim=0)
        out_av[m] = torch.stack([a[m] for a in avs], dim=0)

    out_y = torch.stack(list(ys), dim=0)

    # ensure at least 1 modality available per sample
    all_missing = torch.ones(out_y.shape[0], dtype=torch.bool)
    for m in modalities:
        all_missing = all_missing & (~out_av[m])
    if torch.any(all_missing):
        first_m = next(iter(modalities))
        out_av[first_m][all_missing] = True

    return out_x, out_av, out_y
