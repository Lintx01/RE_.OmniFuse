# OmniFuse (论文代码复现骨架)

本仓库基于论文 *OmniFuse: A general modality fusion framework for multi-modality learning on low-quality medical data*（Information Fusion, 2025）实现一个**可运行的 PyTorch 复现骨架**：

- 缺失模态插补：Cascade Residual Autoencoder (CRA)，含 forward / backward（cycle consistency）
- 不平衡模态融合：Dynamic Weighted Fuse (DWFuse)
- 噪声/懒惰模态激活：Traceable Laziness Activation (TLA)

> 说明：这里提供的是“按论文方法可运行的实现骨架 + 合成数据最小示例”，用于复现方法与训练流程；要复现论文指标（MIMIC / 肺癌数据）需要额外的数据预处理与特定 encoder（BERT/ResNet）接入。

# OmniFuse 复现（本仓库简化实现）

> Windows + VS Code 友好。支持三模态：实验室（总表/单行向量）、超声图像（单病人多张多角度）、文本（病历）。

---

## 1. 快速跑通（使用随机“原始数据”）

### 1.1 生成模拟原始数据（图像+文本+labs总表）
在仓库根目录：

```
python -m omnifuse.tools.make_synth_raw_manifest --out-dir data_synth_raw --n 60
```

会生成：
- `data_synth_raw/manifest.csv`（含 `us` 多张图 `;` 分隔、`text` txt 路径）
- `data_synth_raw/labs_table.csv`（总表）
- `data_synth_raw/us/*.png`
- `data_synth_raw/text/*.txt`

### 1.2 训练 + 验证 + 测试 + 结果落盘（CSV）
（推荐用**纯路径字符串**，不要复制 VS Code 的可点击 `http://_vscodecontentref_...` 链接）

PowerShell：
```
conda activate classical_model
cd "G:\AAAA复现Omnifuse代码"
python -m omnifuse.train_raw_manifest  --manifest "G:\AAAA复现Omnifuse代码\data_synth_raw\manifest.csv" --num-classes 3 --labs-table "G:\AAAA复现Omnifuse代码\data_synth_raw\labs_table.csv" --text-encoder simple --no-pretrained-image --epochs 50 --seed 42 --out-dir "G:\AAAA复现Omnifuse代码\runs\seed42_e10"
```

训练输出目录（可写论文）：
- `runs/.../metrics.csv`：每个 epoch 的 train/val 指标
- `runs/.../test.csv`：最终测试集指标
- `runs/.../confusion_matrix.csv`：测试集混淆矩阵（rows=true, cols=pred）
- `runs/.../split.csv`：train/val/test 样本数
- `runs/.../meta.csv`：超参数/seed/路径

---

## 2. 使用你的真实数据（原始三模态）

### 2.1 manifest.csv（必须）
`manifest.csv` 列必须是：
- `id,label,labs,us,text`

约定：
- `label`：三分类用 `0/1/2`（例如：0=正常，1=轻症，2=重症）
- `us`：单个病人可多张图，使用 `;` 或 `|` 分隔  
  例：`us/0000000001_0.png;us/0000000001_1.png`
- `text`：可以直接写文本，或写 `.txt` 路径
- `labs`：若使用 labs 总表，则此列可留空

### 2.2 labs_table.csv（总表，可选但推荐）
- 至少包含一列 `id`
- 其他列默认都当作数值特征（空/非数值会当 0）
- `id` 会自动做 **10 位纯数字补零**（例如 `123` -> `0000000123`）

训练命令示例：
```
python -m omnifuse.train_raw_manifest ^
  --manifest "D:\your_data\manifest.csv" ^
  --num-classes 3 ^
  --labs-table "D:\your_data\labs_table.csv" ^
  --labs-id-col id ^
  --epochs 50 ^
  --seed 42 ^
  --out-dir "runs\exp_real_seed42"
```

---

## 3. 文本编码器（HF / transformers）

### 3.1 使用 HF BERT（需要 transformers）
命令里把 `--text-encoder hf`，并指定模型：
```
python -m omnifuse.train_raw_manifest ... --text-encoder hf --text-model bert-base-chinese
```

### 3.2 重要：transformers 安装位置与“串包”说明（Windows）
- `transformers` 放用户目录（Roaming）可以，但**不要**在用户目录安装 `numpy/torch/torchvision`（会覆盖 conda 环境导致版本错配）。
- 如果你遇到权限问题（`WinError 5`）无法装进 conda env：可以把 `transformers/tokenizers/sentencepiece` 安装到用户目录：
```
python -m pip install --user "transformers==4.38.2" "tokenizers==0.15.2" sentencepiece safetensors
```

检查用户目录是否有“基础包”（如 numpy）并移除（只保留 transformers 相关）：
```
python -m pip list --user | findstr /i "numpy torch torchvision torchaudio transformers tokenizers sentencepiece"
python -m pip uninstall -y numpy
```

---

## 4. VS Code 复制路径坑（必读）
不要把这种格式复制到终端：
- `[manifest.csv](http://_vscodecontentref_/...)`

终端里必须传“真实路径字符串”，例如：
- `data_synth_raw\manifest.csv`
- `"G:\AAAA复现Omnifuse代码\data_synth_raw\manifest.csv"`

---

## 5. 代码入口
- 原始数据训练（图像+文本+labs总表）：`omnifuse/train_raw_manifest.py`
- 原始数据 Dataset：`omnifuse/datasets_raw.py`
- 模态 encoder：`omnifuse/encoders.py`
- OmniFuse 主体（CRA/DWFuse/TLA）：`omnifuse/model.py`