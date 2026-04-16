# MDS5122 Speech Prediction - Work Log

## Session: 2026-04-15

### Completed Tasks (8/10)

| Task | Description | Status | Commit |
|------|-------------|--------|--------|
| Task 1 | pixi Environment Setup | ✅ Completed | 1a445e7 |
| Task 2 | Unified Codec Wrapper | ✅ Completed | 5f17204 |
| Task 3 | WSJ0 Dataset and Token Cache | ✅ Completed | 022cad6 |
| Task 4 | Codec Language Model | ✅ Completed | 64d15e8 |
| Task 5 | Training Loop | ✅ Completed | 4e8bd34 |
| Task 6 | Inference (Autoregressive Generation) | ✅ Completed | f66862b |
| Task 7 | Evaluation | ✅ Completed | 362aa12 |
| Task 8 | Experiment Configs | ✅ Completed | 4adebce |

### Remaining Tasks (2/10)

| Task | Description | Status |
|------|-------------|--------|
| Task 9 | End-to-End Run | ⏳ In Progress |
| Task 10 | README | ⏳ Pending |

---

## Training Experiments

### EnCodec LM - First Training Run (2026-04-16)

**Configuration:**
- Epochs: 100
- Batch size: 32
- Learning rate: 0.0001
- Weight decay: 0.01
- Dropout: 0.1

**Results:**
```
Epoch   1: train_loss=4.2466, val_loss=3.4837
Epoch   2: train_loss=3.1793, val_loss=3.3463
...
Epoch  97: train_loss=2.3055, val_loss=3.3611
```

**Observations:**
| Metric | Start | End | Change |
|--------|-------|-----|--------|
| train_loss | 4.24 | 2.30 | ↓ 46% ✅ |
| val_loss | 3.48 | 3.34 | ↓ 4% ❌ |

**Problem Identified: Overfitting**
- Training loss dropped significantly (4.2 → 2.3)
- Validation loss barely changed (3.48 → 3.34)
- Gap between train/val loss: ~1.0 (indicates overfitting)

**Root Cause Analysis:**
1. **Data imbalance**: Train set 12,776 samples vs Val set ~400 samples
2. **Insufficient regularization**: Original dropout=0.1, weight_decay=0.01
3. **Model capacity**: 19.7M parameters may be too large for this dataset

**Solution Applied:**
| Parameter | Original | New | Rationale |
|-----------|----------|-----|-----------|
| weight_decay | 0.01 | 0.05 | 5x stronger L2 regularization |
| dropout | 0.1 | 0.2 | 2x more dropout to prevent overfitting |

**Next Steps:**
1. Re-train with new regularization settings
2. Monitor val_loss for improvement
3. Run evaluation after training completes

---

### EnCodec LM - Second Training Run (2026-04-16)

**Configuration (Updated):**
- Epochs: 100
- Batch size: 32
- Learning rate: 0.0001
- **Weight decay: 0.05** (increased from 0.01)
- **Dropout: 0.2** (increased from 0.1)

**Results:**
```
Epoch   1: train_loss=4.3029, val_loss=3.5076
Epoch   2: train_loss=3.2120, val_loss=3.3801
Epoch   3: train_loss=3.1282, val_loss=3.3097
...
Epoch  97: train_loss=2.5505, val_loss=3.1241
Epoch  98: train_loss=2.5512, val_loss=3.1222
Epoch  99: train_loss=2.5494, val_loss=3.1174
Epoch 100: train_loss=2.5511, val_loss=3.1298
Training complete. Best val loss: 3.0551
```

**Comparison with First Run:**
| Metric | Run 1 (低正则化) | Run 2 (高正则化) | 变化 |
|--------|-----------------|-----------------|------|
| Final train_loss | 2.30 | 2.55 | ↑ 0.25 (正则化阻止过拟合) |
| Final val_loss | 3.34 | 3.12 | ↓ 0.22 ✅ |
| Best val_loss | ~3.34 | 3.0551 | ↓ 0.29 ✅ |
| Train-Val Gap | ~1.0 | ~0.5 | ↓ 50% ✅ |

**Key Observations:**
1. **正则化有效**: Train-val gap 从 1.0 缩小到 0.5
2. **Val loss 略有改善**: 从 3.34 降到 3.12 (最佳 3.0551)
3. **但仍存在过拟合**: Val loss 变化幅度小 (~3.5 → ~3.1)

**Possible Reasons for Limited Improvement:**
1. **数据集固有限制**: 验证集只有 ~400 样本，可能不足以代表真实分布
2. **任务本质难度**: 语音 token 预测比文本更难（更少语义约束）
3. **模型架构限制**: 可能需要更多数据或更复杂的架构

**Conclusion:**
- 增加正则化**确实改善了泛化差距**
- 但验证 loss 仍然较高，可能需要：
  - 更大的数据集
  - 不同的模型架构
  - 或者接受这是该任务的固有难度

**Recommendation:**
运行评估脚本查看实际生成质量 (STOI/PESQ/DNSMOS)，这些指标比 loss 更能反映真实效果

---

## Project Structure Created

```
mds5122/
├── pixi.toml                          # Environment configuration
├── src/
│   ├── codec_wrapper.py               # Unified EnCodec/FACodec API
│   ├── dataset.py                     # TokenDataset for LM training
│   ├── model.py                       # CodecLM Transformer decoder
│   ├── train.py                       # Training loop with LR scheduling
│   ├── inference.py                   # Autoregressive speech prediction
│   └── evaluate.py                    # STOI/PESQ/DNSMOS metrics
├── scripts/
│   ├── extract_tokens.py              # Token pre-extraction CLI
│   ├── run_eval.py                    # Full evaluation pipeline
│   └── download_facodec.py            # FACodec checkpoint downloader
├── configs/
│   ├── encodec_lm.yaml                # EnCodec experiment config
│   └── facodec_lm.yaml                # FACodec experiment config
├── token_cache/
│   └── encodec/train/                 # ~12,776 pre-extracted token files
└── docs/
    └── plans/2026-04-15-speech-prediction.md
```

---

## Key Technical Notes

### 1. EnCodec Token Extraction
- Completed for train split (12,776 files)
- Val/test splits extracted

### 2. FACodec Checkpoint
- Downloaded from HuggingFace (Plachta/FAcodec)
- Located at: `FAcodec/checkpoints/FAcodec.pth`

### 3. Model Architecture
- Decoder-only Transformer
- ~19.7M parameters (vocab=1025, d_model=512, 6 layers, 8 heads)
- Weight tying between embedding and output layers

### 4. Training Configuration (Updated)
- Batch size: 32
- Max sequence length: 300 (EnCodec) / 320 (FACodec)
- Learning rate: 0.0001 with warmup + cosine decay
- Gradient clipping: 1.0
- **Weight decay: 0.05** (increased from 0.01)
- **Dropout: 0.2** (increased from 0.1)

---

## Bug Fixes Applied

| Issue | Fix | Commit |
|-------|-----|--------|
| FACodec encode() error | Use content_quantizer directly | 4af3f42 |
| YAML lr parsed as string | Changed 1e-4 to 0.0001 | ed69d28 |
| No checkpoint resume support | Added resume parameter | 788a4ac |

---

## Git Status

- Branch: `feature/speech-prediction`
- Remote: `origin/feature/speech-prediction`
