# MDS5122 Speech Prediction

基于神经语音编解码器（Neural Audio Codec）的语音信号自回归预测项目。

## 项目简介

本项目使用 EnCodec 或 FACodec 将语音信号转换为离散 token 序列，训练 Transformer 语言模型学习 token 序列的统计规律，实现语音信号的自回归预测。

**核心流程**：
```
语音信号 → Codec编码 → Token序列 → LM训练 → Token预测 → Codec解码 → 预测语音
```

## 功能特性

- 支持 EnCodec (Meta) 和 FACodec 两种神经语音编解码器
- GPT-2 风格的 Decoder-only Transformer 语言模型
- 完整的训练流程：学习率预热、余弦退火、梯度裁剪
- 支持从 checkpoint 恢复训练
- 多维度评估指标：STOI、PESQ、DNSMOS
- TensorBoard 可视化支持

## 项目结构

```
mds5122/
├── configs/
│   ├── encodec_lm.yaml      # EnCodec 实验配置
│   └── facodec_lm.yaml      # FACodec 实验配置
├── src/
│   ├── codec_wrapper.py     # EnCodec/FACodec 统一接口
│   ├── dataset.py           # Token 数据集
│   ├── model.py             # Transformer 语言模型
│   ├── train.py             # 训练循环
│   ├── inference.py         # 自回归生成推理
│   └── evaluate.py          # STOI/PESQ/DNSMOS 评估
├── scripts/
│   ├── extract_tokens.py    # Token 预提取
│   ├── run_eval.py          # 完整评估流程
│   └── download_facodec.py  # FACodec 模型下载
├── checkpoints/             # 模型保存目录
├── token_cache/             # 预提取的 token 缓存
├── results/                 # 评估结果
├── docs/
│   ├── WORK_LOG.md          # 工作日志
│   ├── OPERATION_GUIDE.md   # 详细操作指南
│   └── EXPERIMENT_REPORT.md # 实验报告
├── pixi.toml                # 环境配置
└── README.md
```

## 环境要求

- Python 3.10
- PyTorch 2.3+
- CUDA 12.1 (GPU 训练)
- pixi (包管理器)

## 安装

```bash
# 克隆项目
git clone <repo-url>
cd mds5122

# 安装 pixi (如果未安装)
curl -fsSL https://pixi.sh/install.sh | bash

# 安装依赖
pixi install
```

## 快速开始

### 1. 准备数据集

将 WSJ0 数据集放置在 `dataset/wsj0/` 目录下，或修改配置文件中的 `wsj0_root` 路径。

### 2. 下载 FACodec 模型 (可选)

```bash
pixi run python scripts/download_facodec.py
```

### 3. 提取 Token

```bash
# EnCodec token 提取
pixi run python scripts/extract_tokens.py \
    --codec encodec \
    --wsj0_root /path/to/wsj0 \
    --split all \
    --device cuda

# FACodec token 提取
pixi run python scripts/extract_tokens.py \
    --codec facodec \
    --wsj0_root /path/to/wsj0 \
    --split all \
    --device cuda \
    --facodec_repo ./FAcodec \
    --facodec_ckpt ./FAcodec/checkpoints/FAcodec.pth \
    --facodec_cfg ./FAcodec/configs/config.yml
```

### 4. 训练模型

```bash
# 训练 EnCodec LM
pixi run python src/train.py --config configs/encodec_lm.yaml

# 训练 FACodec LM
pixi run python src/train.py --config configs/facodec_lm.yaml
```

### 5. 评估模型

```bash
# 评估 EnCodec LM
pixi run python scripts/run_eval.py \
    --config configs/encodec_lm.yaml \
    --split test \
    --out_json results/encodec_test.json

# 评估 FACodec LM
pixi run python scripts/run_eval.py \
    --config configs/facodec_lm.yaml \
    --split test \
    --out_json results/facodec_test.json
```

## 模型架构

**CodecLM**: GPT-2 风格 Decoder-only Transformer

| 参数 | 值 |
|------|-----|
| vocab_size | 1026 (1024 codec tokens + BOS + PAD) |
| d_model | 512 |
| n_heads | 8 |
| n_layers | 6 |
| ffn_dim | 2048 |
| dropout | 0.2 |
| **总参数量** | **21.79M** |

**架构特点**:
- Pre-LayerNorm (GPT-2 风格)
- 因果自注意力 (Causal Self-Attention)
- Weight Tying (embedding 与 output 层共享权重)
- 顶层-k 采样生成

## 训练配置

| 参数 | 值 |
|------|-----|
| batch_size | 32 |
| max_len | 300 tokens (~4 sec) |
| epochs | 100 |
| learning_rate | 0.0001 |
| weight_decay | 0.05 |
| warmup_steps | 500 |
| optimizer | AdamW |
| LR schedule | Cosine decay with warmup |

## 实验结果

详细结果见 `docs/EXPERIMENT_REPORT.md`。

### 主要发现

1. **感知质量可接受**: PESQ 和 DNSMOS 指标表明生成的语音在感知上自然
2. **数据量不足是主要瓶颈**: 当前数据量仅为推荐量的 1/30 ~ 1/300
3. **FACodec 重建质量较低**: 只使用 content codebook，丢失了韵律和音色信息

## 可用任务

项目预定义了以下 pixi 任务：

```bash
pixi run extract-encodec   # 提取 EnCodec tokens
pixi run extract-facodec   # 提取 FACodec tokens
pixi run train-encodec     # 训练 EnCodec LM
pixi run train-facodec     # 训练 FACodec LM
pixi run eval-encodec      # 评估 EnCodec LM
pixi run eval-facodec      # 评估 FACodec LM
```

## 监控训练

```bash
# 启动 TensorBoard
pixi run tensorboard --logdir checkpoints/encodec_lm/tb --port 6006

# 浏览器访问
# http://localhost:6006
```

## 从中断恢复训练

训练脚本自动检测并加载最新的 checkpoint：

```bash
# 直接重新运行训练命令即可自动恢复
pixi run python src/train.py --config configs/encodec_lm.yaml
```

## 参考文献

- [EnCodec](https://github.com/facebookresearch/encodec) - Meta AI
- [FAcodec](https://github.com/plachta1/FAcodec) - Flow-based Attribute Decoupling
- [WSJ0](https://catalog.ldc.upenn.edu/LDC93s6a) - Wall Street Journal Corpus

## 许可证

本项目仅供学术研究使用。

---

*MDS5122 Final Project - 2026*
