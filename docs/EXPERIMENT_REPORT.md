# MDS5122 语音预测实验报告

## 一、实验概述

### 1.1 任务描述
使用神经语音编解码器（Neural Audio Codec）将语音信号转换为离散 token 序列，训练语言模型学习 token 序列的统计规律，实现语音信号的自回归预测。

### 1.2 实验流程
```
语音信号 → Codec编码 → Token序列 → LM训练 → Token预测 → Codec解码 → 预测语音
```

### 1.3 核心方法
- 给定语音前半部分作为 context
- 使用 LM 自回归预测后半部分的 token
- 解码生成预测的语音信号

---

## 二、实验设置

### 2.1 数据集
| 数据集 | 来源 | 样本数 |
|--------|------|--------|
| 训练集 | WSJ0 si_tr_s | 12,776 |
| 验证集 | WSJ0 si_dt_05 | 1,206 |
| 测试集 | WSJ0 si_et_05 | 651 |

**音频统计**:
- 采样率: 16 kHz (原始) → 24 kHz (重采样)
- 平均时长: ~7 秒
- 平均 token 长度: ~512 tokens
- 总训练 tokens: ~6.76M

### 2.2 语音编解码器

#### EnCodec (Meta)
- 模型: `facebook/encodec_24khz`
- 帧率: 75 fps (每帧 320 样本)
- Codebook: 1024 个 token
- 特点: 8 个 residual codebooks，取第一个

#### FACodec
- 模型: `Plachta/FAcodec`
- 帧率: 75 fps
- Codebook: 1024 个 token  
- 特点: 1 个 content codebook，分离 prosody/timbre

### 2.3 语言模型

**架构**: GPT-2 风格 Decoder-only Transformer

| 参数 | 值 |
|------|-----|
| vocab_size | 1026 (1024 + BOS + PAD) |
| d_model | 512 |
| n_heads | 8 |
| n_layers | 6 |
| ffn_dim | 2048 |
| dropout | 0.2 |
| **总参数量** | **21.79M** |

### 2.4 训练配置

| 参数 | 值 |
|------|-----|
| batch_size | 32 |
| max_len | 300 tokens (~4 sec) |
| epochs | 100 |
| learning_rate | 0.0001 |
| weight_decay | 0.05 |
| warmup_steps | 500 |
| label_smoothing | 0.1 |
| optimizer | AdamW |
| LR schedule | Cosine decay with warmup |

### 2.5 推理配置

| 参数 | 值 |
|------|-----|
| temperature | 0.8 |
| top_k | 100 |
| max_context | 149 tokens |
| max_generation | 150 tokens |

---

## 三、训练结果

### 3.1 EnCodec LM

| 指标 | 初始 | 最终 | 最佳 |
|------|------|------|------|
| Train Loss | 5.3912 | 3.2891 | - |
| Val Loss | 4.0901 | 3.7894 | 3.7201 @ epoch 33 |

**收敛分析**:
- 训练 loss 下降 39%
- 验证 loss 下降 8%
- Train-Val gap: ~0.43 (表明轻度过拟合)
- 最佳模型在 epoch 33

### 3.2 FACodec LM

| 指标 | 初始 | 最终 | 最佳 |
|------|------|------|------|
| Train Loss | 6.4347 | 3.7169 | - |
| Val Loss | 4.9712 | 4.0195 | 4.0054 @ epoch 46 |

**收敛分析**:
- 训练 loss 下降 42%
- 验证 loss 下降 19%
- Train-Val gap: ~0.30
- 最佳模型在 epoch 46

---

## 四、评估结果

### 4.1 主要结果

```
┌──────────────────┬────────┬────────┬────────┐
│                  │  STOI  │  PESQ  │ DNSMOS │
├──────────────────┼────────┼────────┼────────┤
│ EnCodec Baseline │ 0.6724 │ 1.2004 │ 2.5103 │
│ EnCodec LM       │ 0.2842 │ 1.2849 │ 2.4701 │
├──────────────────┼────────┼────────┼────────┤
│ FACodec Baseline │ 0.3330 │ 1.0838 │ 2.3395 │
│ FACodec LM       │ 0.3487 │ 1.0559 │ 2.3201 │
└──────────────────┴────────┴────────┴────────┘
```

**注**: 
- Baseline = 编码真实目标音频后解码 (测试 codec 重建能力)
- LM = 编码前半部分，用 LM 预测后半部分 token 后解码 (测试预测能力)
- 两者测试的是不同任务，STOI 不直接可比

### 4.2 STOI 分布统计

**EnCodec LM**:
```
Mean:   0.2842
Std:    0.1498
Min:    -0.0814
Max:    0.6406
Median: 0.2893
Negative count: 14/651 (2.1%)
```

**FACodec LM**:
```
Mean:   0.3487
Std:    0.1402
Min:    -0.0293
Max:    0.5719
Median: 0.3861
Negative count: 2/651 (0.3%)
```

### 4.3 评估指标说明

| 指标 | 全称 | 范围 | 说明 |
|------|------|------|------|
| STOI | Short-Time Objective Intelligibility | 0~1 | 时域信号相似度，与可懂度相关 |
| PESQ | Perceptual Evaluation of Speech Quality | -0.5~4.5 | 感知语音质量 |
| DNSMOS | Deep Noise Suppression MOS | 1~5 | 基于深度学习的语音质量评估 |

---

## 五、结果分析

### 5.1 主要发现

1. **EnCodec LM 感知质量可接受**
   - PESQ 1.28 > Baseline 1.20 (略有提升)
   - DNSMOS 2.47 ≈ Baseline 2.51 (基本持平)
   - 说明生成的语音在感知上是自然的

2. **EnCodec LM 时域相似度低**
   - STOI 0.28 远低于 Baseline 0.67
   - 主要原因是预测任务的固有难度 (vs 重建任务)
   - 14 个样本 STOI 为负，表示预测方向错误

3. **FACodec 本身重建质量差**
   - Baseline STOI 仅 0.33 (vs EnCodec 0.67)
   - 只用 1 个 content codebook，丢失了 prosody/timbre
   - LM 在此基础上 STOI 略有提升 (0.35 > 0.33)

4. **FACodec LM 方差更小**
   - STOI 标准差 0.14 (vs EnCodec 0.15)
   - 负样本更少 (2 vs 14)
   - 可能因为 FACodec content token 信息更集中

### 5.2 效果瓶颈分析

**数据量不足** (核心瓶颈):
```
当前: 21.79M 参数, 6.76M tokens
比例: 0.31 tokens/params

参考标准:
- GPT-2: 68 tokens/params  
- 推荐: 10-100 tokens/params
```

当前数据量仅为推荐量的 **1/30 ~ 1/300**。

### 5.3 与 Baseline 的区别

| 对比项 | Baseline | LM |
|--------|----------|-----|
| 输入 | 真实后半部分音频 | 前半部分音频编码 |
| 任务 | Codec 重建 | 自回归预测 |
| 信息量 | 完整目标信息 | 只有上下文信息 |
| 难度 | 低 (测试 codec 能力) | 高 (测试预测能力) |

---

## 六、项目结构

```
mds5122/
├── configs/
│   ├── encodec_lm.yaml     # EnCodec 实验配置
│   └── facodec_lm.yaml     # FACodec 实验配置
├── src/
│   ├── codec_wrapper.py    # EnCodec/FACodec 统一接口
│   ├── dataset.py          # Token 数据集
│   ├── model.py            # Transformer 语言模型
│   ├── train.py            # 训练循环
│   ├── inference.py        # 自回归生成推理
│   └── evaluate.py         # STOI/PESQ/DNSMOS 评估
├── scripts/
│   ├── extract_tokens.py   # Token 预提取
│   └── run_eval.py         # 完整评估流程
├── checkpoints/
│   ├── encodec_lm/         # EnCodec LM 模型
│   └── facodec_lm/         # FACodec LM 模型
├── results/
│   ├── encodec_test_v2.json
│   └── facodec_test_v2.json
└── docs/
    └── OPERATION_GUIDE.md  # 操作指南
```

---

## 七、结论

### 7.1 实验成果
1. 实现了完整的语音预测流程 (编码→训练→预测→评估)
2. 在有限数据条件下，模型学到了有效的 token 分布
3. 生成的语音在感知质量上可接受 (PESQ/DNSMOS)

### 7.2 局限性
1. 数据量严重不足，限制了模型性能
2. STOI 时域相似度低，预测准确性有提升空间
3. 部分样本预测方向错误 (STOI 为负)

### 7.3 改进方向
1. **增加数据**: 使用更大规模的语音数据集
2. **缩小模型**: 降低参数量以匹配当前数据规模
3. **改进架构**: 尝试更适合序列预测的模型结构
4. **优化训练**: 调整学习率、数据增强等策略

---

## 附录: 运行命令

### Token 提取
```bash
PIXI="env -u HTTP_PROXY -u HTTPS_PROXY /home/franka/.pixi/bin/pixi run"

# EnCodec
$PIXI python scripts/extract_tokens.py \
    --codec encodec \
    --wsj0_root /home/franka/Development/mds5122/dataset/wsj0 \
    --out_dir /home/franka/Development/mds5122/token_cache \
    --split all --device cuda

# FACodec
$PIXI python scripts/extract_tokens.py \
    --codec facodec \
    --wsj0_root /home/franka/Development/mds5122/dataset/wsj0 \
    --out_dir /home/franka/Development/mds5122/token_cache \
    --split all --device cuda \
    --facodec_repo /home/franka/Development/mds5122/FAcodec \
    --facodec_ckpt /home/franka/Development/mds5122/FAcodec/checkpoints/FAcodec.pth \
    --facodec_cfg /home/franka/Development/mds5122/FAcodec/configs/config.yml
```

### 模型训练
```bash
$PIXI python src/train.py --config configs/encodec_lm.yaml
$PIXI python src/train.py --config configs/facodec_lm.yaml
```

### 模型评估
```bash
mkdir -p results
$PIXI python scripts/run_eval.py --config configs/encodec_lm.yaml --split test --baseline --out_json results/encodec_test_v2.json
$PIXI python scripts/run_eval.py --config configs/facodec_lm.yaml --split test --baseline --out_json results/facodec_test_v2.json
```

---

*报告生成时间: 2026-04-17*
