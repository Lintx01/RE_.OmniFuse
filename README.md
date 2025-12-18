# OmniFuse (论文代码复现骨架)

本仓库基于论文 *OmniFuse: A general modality fusion framework for multi-modality learning on low-quality medical data*（Information Fusion, 2025）实现一个**可运行的 PyTorch 复现骨架**：

- 缺失模态插补：Cascade Residual Autoencoder (CRA)，含 forward / backward（cycle consistency）
- 不平衡模态融合：Dynamic Weighted Fuse (DWFuse)
- 噪声/懒惰模态激活：Traceable Laziness Activation (TLA)

> 说明：这里提供的是“按论文方法可运行的实现骨架 + 合成数据最小示例”，用于复现方法与训练流程；要复现论文指标（MIMIC / 肺癌数据）需要额外的数据预处理与特定 encoder（BERT/ResNet）接入。

## 快速开始（合成数据）

1) 安装依赖

```bash
pip install -r requirements.txt
```

2) 运行最小训练

```bash
python -m omnifuse.train_synth
```

你会看到 loss 下降与训练日志，证明 forward/backward imputation、DWFuse、TLA 的训练目标可以端到端跑通。

## 接入你自己的三模态数据（实验室/超声/病历）

为了先把流程跑通，这里提供一种**不依赖 BERT/ResNet** 的接入方式：你把每个模态先转成一个定长向量特征，保存为 `.npy`（1D float32），然后用一个 CSV 作为索引。

### 1) 准备 manifest.csv

CSV 需要列名：`id,label,labs,us,text`

- `labs`：血常规/尿常规等实验室特征向量（例如 1×D 的 `.npy`）
- `us`：超声影像的特征向量（你可以用任意方法先抽特征）
- `text`：病历文本特征向量（同理可先抽特征）
- 某个模态缺失时，该单元格留空即可

### 2) 训练

```bash
python -m omnifuse.train_manifest --manifest path\\to\\manifest.csv --num-classes 2
```

### 3) 下一步（可选）

如果你希望“直接从原始超声图像/原始文本”端到端训练，我可以继续加：
- 超声：`torchvision` + ResNet encoder
- 文本：`transformers` + BERT encoder

需要你提供：任务类型、标签定义、以及文件组织结构。

## 端到端：从原始数据训练（超声图像 + 病历文本 + 实验室数值）

当你的三模态是原始数据时，建议用这个入口。依赖单独放在 `requirements_raw.txt`：

```bash
pip install -r requirements_raw.txt
```

### manifest.csv 格式

列名固定：`id,label,labs,us,text`

- `id`：会在读取时对**纯数字 ID** 自动做 10 位归一化：不足 10 位左侧补 `0`（`zfill(10)`）。

- `label`：三分类建议用 `0=正常, 1=轻症, 2=重症`
- `labs` 支持三种写法（二选一即可）：
	- 直接写 JSON 向量：`[0.1, 2.3, ...]`
	- 写 `.npy` 路径：`labs/xxx.npy`（1D float32）
	- 写单行数值 `.csv` 路径：`labs/xxx.csv`
- `us`：超声图像文件路径（png/jpg/bmp/tif 等），一个病人多角度多张图像可用分隔符 `;` 或 `|` 填多个路径，例如：
	- `us/a.png;us/b.png;us/c.png`
	- 缺失留空
- `text`：可以直接写文本内容，或写一个 `.txt` 文件路径，缺失留空

### 训练

```bash
python -m omnifuse.train_raw_manifest --manifest path\\to\\manifest.csv --num-classes 3
```

如果你的实验室数据是“总表”（一张 CSV，包含 `id` 列和多列数值特征），可以额外传入：

```bash
python -m omnifuse.train_raw_manifest --manifest path\\to\\manifest.csv --num-classes 3 \
	--labs-table path\\to\\labs_table.csv --labs-id-col id
```

可选参数：

```bash
python -m omnifuse.train_raw_manifest --manifest path\\to\\manifest.csv --num-classes 3 \
	--text-model bert-base-chinese --image-backbone resnet34 --image-size 224 --freeze-text
```

## 代码结构

- `omnifuse/model.py`：OmniFuse 主模型（encoder + CRA 插补 + DWFuse + TLA）
- `omnifuse/modules.py`：CRA / Residual Autoencoder / DWFuse 等组件
- `omnifuse/train_synth.py`：合成数据训练脚本

## 下一步

如果你希望“复现这篇文章的代码”指的是**对接论文的真实设置**（EHR=76 维时序、CXR=ResNet34、REP=BERT），我可以继续把：
- EHR/CXR/REP 的 encoder 接口对齐论文描述
- 数据加载与缺失模式采样
- 训练/验证 AUROC 计算

你只需要告诉我：你要复现的是哪一个任务（MIMIC in-hospital mortality / phenotyping / 肺癌分型）以及数据在本机的路径结构。
