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
| Task 9 | End-to-End Run | ⏳ Pending |
| Task 10 | README | ⏳ Pending |

### Project Structure Created

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

### Key Implementation Notes

1. **EnCodec Token Extraction**: Completed for train split (12,776 files)
   - Val/test splits need extraction: `pixi run extract-encodec --split val` and `--split test`

2. **FACodec Checkpoint**: Downloaded from HuggingFace (Plachta/FAcodec)
   - Located at: `FAcodec/checkpoints/FAcodec.pth`

3. **Model Architecture**: Decoder-only Transformer
   - ~19.7M parameters (vocab=1025, d_model=512, 6 layers, 8 heads)
   - Weight tying between embedding and output layers

4. **Training Configuration**:
   - Batch size: 32
   - Max sequence length: 300 (EnCodec) / 320 (FACodec)
   - Learning rate: 1e-4 with warmup + cosine decay
   - Gradient clipping: 1.0

### Next Steps

1. **Task 9 - End-to-End Run**:
   - Extract val/test tokens for EnCodec
   - Train EnCodec LM
   - Extract all FACodec tokens
   - Train FACodec LM
   - Run evaluation and compare results

2. **Task 10 - README**:
   - Document setup instructions
   - Document reproduction steps
   - Add bonus task description

### Verification Status

All implemented modules have passed:
- ✅ Spec compliance review
- ✅ Code quality review

### Git Status

- Branch: `feature/speech-prediction`
- Ready to push to `origin`
