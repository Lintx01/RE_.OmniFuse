from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class RawManifestRow:
    sample_id: str
    label: int
    labs: str
    us: str
    text: str


def normalize_id(value: str, width: int = 10) -> str:
    """Normalize ID to a fixed width for matching.

    Requirement: keep 10 digits; if shorter, left-pad with 0.
    We only apply zero-padding when the ID is purely numeric.
    """
    s = str(value).strip()
    if s.isdigit() and len(s) < width:
        return s.zfill(width)
    if s.isdigit() and len(s) > width:
        raise ValueError(f"ID 长度超过 {width} 位：{s}（请检查是否需要截断/清洗）")
    return s


def _is_missing(v: str) -> bool:
    v = (v or "").strip()
    return v == "" or v.lower() in {"none", "null", "nan"}


def _abs_path(base_dir: str, maybe_path: str) -> str:
    p = maybe_path
    if not os.path.isabs(p):
        p = os.path.join(base_dir, p)
    return p


def _parse_labs_vector(value: str, base_dir: str) -> Tuple[np.ndarray, bool]:
    """Support labs as:
    - inline JSON list: "[0.1, 2.3, ...]"
    - .npy path
    - .csv path with a single row of numeric values (no header)
    """
    if _is_missing(value):
        return np.zeros((0,), dtype=np.float32), False

    v = value.strip()
    if v.startswith("["):
        arr = np.array(json.loads(v), dtype=np.float32)
        return arr, True

    path = _abs_path(base_dir, v)
    if path.lower().endswith(".npy"):
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 1:
            raise ValueError(f"labs 期望 1D 向量：{path} shape={arr.shape}")
        return arr, True

    if path.lower().endswith(".csv"):
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            row = next(reader)
        arr = np.array([float(x) for x in row], dtype=np.float32)
        return arr, True

    raise ValueError("labs 列仅支持 JSON 向量、.npy 或单行数值 .csv")


def _load_text(value: str, base_dir: str) -> Tuple[str, bool]:
    if _is_missing(value):
        return "", False

    v = value.strip()
    # treat as path if file exists
    p = _abs_path(base_dir, v)
    if os.path.exists(p) and os.path.isfile(p):
        with open(p, "r", encoding="utf-8") as f:
            return f.read(), True
    return v, True


class RawTriModalManifestDataset(Dataset):
    """端到端三模态 Dataset（原始数据）。

    manifest.csv 列：id,label,labs,us,text

    - labs: JSON 向量 / .npy / 单行数值 .csv（可空）
    - us: 超声图像路径（png/jpg/bmp/tif... 可空）
    - text: 文本内容或 txt 文件路径（可空）

    缺失：对应单元格留空。
    """

    def __init__(
        self,
        manifest_csv: str,
        image_size: int = 224,
        max_us_images: int = 8,
        labs_table_csv: Optional[str] = None,
        labs_id_col: str = "id",
        labs_feature_cols: Optional[List[str]] = None,
    ):
        super().__init__()
        self.manifest_csv = manifest_csv
        self.base_dir = os.path.dirname(os.path.abspath(manifest_csv))
        self.image_size = image_size
        self.max_us_images = max_us_images

        self._labs_table: Optional[Dict[str, np.ndarray]] = None
        if labs_table_csv is not None:
            labs_table_path = labs_table_csv
            if not os.path.isabs(labs_table_path):
                labs_table_path = os.path.join(self.base_dir, labs_table_path)
            self._labs_table, self.labs_dim = self._load_labs_table(
                labs_table_path, labs_id_col=labs_id_col, labs_feature_cols=labs_feature_cols
            )

        rows: List[RawManifestRow] = []
        with open(manifest_csv, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            required = {"id", "label", "labs", "us", "text"}
            if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
                raise ValueError(f"manifest 缺少列：需要 {sorted(required)}，实际 {reader.fieldnames}")
            for r in reader:
                rows.append(
                    RawManifestRow(
                        sample_id=normalize_id(r["id"]),
                        label=int(r["label"]),
                        labs=str(r.get("labs", "")),
                        us=str(r.get("us", "")),
                        text=str(r.get("text", "")),
                    )
                )
        if len(rows) == 0:
            raise ValueError("manifest 为空")
        self.rows = rows

        # infer labs dim
        if self._labs_table is None:
            labs_dim = None
            for row in self.rows:
                vec, ok = _parse_labs_vector(row.labs, self.base_dir)
                if ok and vec.shape[0] > 0:
                    labs_dim = int(vec.shape[0])
                    break
            if labs_dim is None:
                raise ValueError("labs 在整个数据集中都缺失，无法训练（可考虑提供 labs_table_csv）")
            self.labs_dim = labs_dim

        try:
            from PIL import Image  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise RuntimeError("需要安装 pillow 才能读取超声图像") from e

        try:
            import torchvision.transforms as T
        except Exception as e:  # pragma: no cover
            raise RuntimeError("需要安装 torchvision 才能使用图像 transforms") from e

        self.image_tf = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                # ImageNet normalization (works fine as a default)
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _split_paths(value: str) -> List[str]:
        v = (value or "").strip()
        if _is_missing(v):
            return []
        parts: List[str] = []
        for chunk in v.replace("|", ";").replace(",", ";").split(";"):
            p = chunk.strip()
            if p:
                parts.append(p)
        return parts

    def _load_images(self, value: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load up to K ultrasound images for a sample.

        Returns:
        - images: FloatTensor [K, 3, H, W]
        - mask: BoolTensor [K] indicating valid slots
        """
        paths = self._split_paths(value)
        K = self.max_us_images
        images = torch.zeros(K, 3, self.image_size, self.image_size)
        mask = torch.zeros(K, dtype=torch.bool)

        if len(paths) == 0:
            return images, mask

        from PIL import Image

        for i, rel in enumerate(paths[:K]):
            path = _abs_path(self.base_dir, rel)
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            img = Image.open(path).convert("RGB")
            images[i] = self.image_tf(img)
            mask[i] = True

        return images, mask

    @staticmethod
    def _load_labs_table(
        labs_table_csv: str, labs_id_col: str, labs_feature_cols: Optional[List[str]]
    ) -> Tuple[Dict[str, np.ndarray], int]:
        table: Dict[str, np.ndarray] = {}
        with open(labs_table_csv, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("labs_table_csv 没有表头")
            if labs_id_col not in reader.fieldnames:
                raise ValueError(f"labs_id_col={labs_id_col} 不在 labs_table_csv 表头里：{reader.fieldnames}")

            feature_cols = labs_feature_cols
            if feature_cols is None:
                feature_cols = [c for c in reader.fieldnames if c != labs_id_col]
            if len(feature_cols) == 0:
                raise ValueError("labs_table_csv 没有可用的数值特征列")

            dim = len(feature_cols)
            for r in reader:
                sid = normalize_id(r[labs_id_col])
                if sid == "":
                    continue
                vals = []
                for c in feature_cols:
                    raw = (r.get(c, "") or "").strip()
                    if raw == "":
                        vals.append(0.0)
                    else:
                        try:
                            vals.append(float(raw))
                        except ValueError:
                            vals.append(0.0)
                table[sid] = np.array(vals, dtype=np.float32)
        if len(table) == 0:
            raise ValueError("labs_table_csv 解析后为空")
        return table, dim

    def __getitem__(self, idx: int):
        row = self.rows[idx]

        # labs: prefer per-row spec; otherwise from total table
        labs_vec, labs_ok = _parse_labs_vector(row.labs, self.base_dir)
        if not labs_ok and self._labs_table is not None:
            vec = self._labs_table.get(row.sample_id)
            if vec is not None:
                labs_vec = vec
                labs_ok = True

        if labs_ok and labs_vec.shape[0] != self.labs_dim:
            raise ValueError(f"labs 维度不一致：期望 {self.labs_dim} 实际 {labs_vec.shape[0]} (id={row.sample_id})")
        if not labs_ok:
            labs_vec = np.zeros((self.labs_dim,), dtype=np.float32)

        us_imgs, us_mask = self._load_images(row.us)
        us_ok = bool(us_mask.any().item())
        text_str, text_ok = _load_text(row.text, self.base_dir)

        x = {
            "labs": torch.from_numpy(labs_vec.astype(np.float32)),
            "us": us_imgs,
            "us_mask": us_mask,
            "text": text_str,  # keep as python string, encoder will tokenize
        }
        available = {
            "labs": torch.tensor(bool(labs_ok), dtype=torch.bool),
            "us": torch.tensor(bool(us_ok), dtype=torch.bool),
            "text": torch.tensor(bool(text_ok), dtype=torch.bool),
        }
        y = torch.tensor(row.label, dtype=torch.long)
        return x, available, y


def collate_raw_trifuse(batch):
    xs, avs, ys = zip(*batch)

    out_x: Dict[str, object] = {}
    out_av: Dict[str, torch.Tensor] = {}

    out_x["labs"] = torch.stack([x["labs"] for x in xs], dim=0)
    out_x["us"] = torch.stack([x["us"] for x in xs], dim=0)  # [B, K, 3, H, W]
    out_x["us_mask"] = torch.stack([x["us_mask"] for x in xs], dim=0)  # [B, K]
    out_x["text"] = [x["text"] for x in xs]

    out_av["labs"] = torch.stack([a["labs"] for a in avs], dim=0)
    out_av["us"] = torch.stack([a["us"] for a in avs], dim=0)
    out_av["text"] = torch.stack([a["text"] for a in avs], dim=0)

    out_y = torch.stack(list(ys), dim=0)

    # ensure at least 1 modality available per sample
    all_missing = (~out_av["labs"]) & (~out_av["us"]) & (~out_av["text"])
    if torch.any(all_missing):
        out_av["labs"][all_missing] = True

    return out_x, out_av, out_y
