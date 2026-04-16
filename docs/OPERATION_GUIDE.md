# MDS5122 语音预测项目 - 操作指南

## 剩余工作概览

| 任务 | 状态 | 说明 |
|------|------|------|
| Task 9: End-to-End Run | ⏳ 需要手动执行 | 运行训练和评估（约需 6-12 小时） |
| Task 10: README | ⏳ 待编写 | 可在训练期间完成 |

---

## 一、环境准备

```bash
cd /home/franka/Development/mds5122

# 确保环境已安装（已完成）
pixi install

# 验证 CUDA
pixi run python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
# 预期输出: PyTorch: 2.3.x, CUDA: True
```

---

## 二、Token 提取（必须先完成）

### 2.1 EnCodec Token 提取

**已完成**: train split (12,776 文件)

**待执行**: val 和 test split

```bash
# 提取 val split（约 5 分钟）
pixi run python scripts/extract_tokens.py \
    --codec encodec \
    --wsj0_root /home/franka/Development/mds5122/wsj0 \
    --out_dir /home/franka/Development/mds5122/token_cache \
    --split val

# 提取 test split（约 5 分钟）
pixi run python scripts/extract_tokens.py \
    --codec encodec \
    --wsj0_root /home/franka/Development/mds5122/wsj0 \
    --out_dir /home/franka/Development/mds5122/token_cache \
    --split test
```

**验证**:
```bash
# 检查文件数量
ls token_cache/encodec/train/*.pt | wc -l  # 应约 12776
ls token_cache/encodec/val/*.pt | wc -l    # 应约 400
ls token_cache/encodec/test/*.pt | wc -l   # 应约 400
```

### 2.2 FACodec Token 提取

```bash
# 提取所有 split（约 30 分钟）
pixi run python scripts/extract_tokens.py \
    --codec facodec \
    --wsj0_root /home/franka/Development/mds5122/wsj0 \
    --out_dir /home/franka/Development/mds5122/token_cache \
    --split all \
    --facodec_repo /home/franka/Development/mds5122/FAcodec \
    --facodec_ckpt /home/franka/Development/mds5122/FAcodec/checkpoints/FAcodec.pth \
    --facodec_cfg /home/franka/Development/mds5122/FAcodec/configs/config.yml
```

**验证**:
```bash
ls token_cache/facodec/train/*.pt | wc -l
ls token_cache/facodec/val/*.pt | wc -l
ls token_cache/facodec/test/*.pt | wc -l
```

---

## 三、模型训练

### 3.1 EnCodec LM 训练

```bash
# 训练 EnCodec 语言模型（约 4-6 小时）
pixi run python src/train.py --config configs/encodec_lm.yaml
```

**监控训练**:
```bash
# 另开终端查看 TensorBoard
pixi run tensorboard --logdir checkpoints/encodec_lm/tb --port 6006

# 浏览器访问: http://localhost:6006
```

**预期结果**:
- 训练 loss: ~7 → <5（约 20 epochs）
- 验证 loss 应持续下降
- 最佳模型保存在: `checkpoints/encodec_lm/best.pt`

### 3.2 FACodec LM 训练

```bash
# 训练 FACodec 语言模型（约 4-6 小时）
pixi run python src/train.py --config configs/facodec_lm.yaml
```

**监控训练**:
```bash
pixi run tensorboard --logdir checkpoints/facodec_lm/tb --port 6007
```

---

## 四、模型评估

### 4.1 评估 EnCodec 模型

```bash
# 创建结果目录
mkdir -p results

# 评估 EnCodec
pixi run python scripts/run_eval.py \
    --config configs/encodec_lm.yaml \
    --split test \
    --out_json results/encodec_test.json
```

### 4.2 评估 FACodec 模型

```bash
# 评估 FACodec
pixi run python scripts/run_eval.py \
    --config configs/facodec_lm.yaml \
    --split test \
    --out_json results/facodec_test.json
```

### 4.3 比较结果

```bash
pixi run python -c "
import json
for name, path in [('EnCodec', 'results/encodec_test.json'),
                   ('FACodec', 'results/facodec_test.json')]:
    s = json.load(open(path))['summary']
    print(f'{name}: STOI={s[\"stoi\"]:.4f}, PESQ={s[\"pesq\"]:.4f}, DNSMOS={s[\"dnsmos\"]:.4f}')
"
```

---

## 五、完整执行流程（一键运行）

### 方案 A: 顺序执行

```bash
#!/bin/bash
# 保存为 run_all.sh

set -e

echo "=== Step 1: Extract EnCodec tokens (val/test) ==="
pixi run python scripts/extract_tokens.py --codec encodec --split val
pixi run python scripts/extract_tokens.py --codec encodec --split test

echo "=== Step 2: Extract FACodec tokens (all) ==="
pixi run python scripts/extract_tokens.py \
    --codec facodec --split all \
    --facodec_repo /home/franka/Development/mds5122/FAcodec \
    --facodec_ckpt /home/franka/Development/mds5122/FAcodec/checkpoints/FAcodec.pth \
    --facodec_cfg /home/franka/Development/mds5122/FAcodec/configs/config.yml

echo "=== Step 3: Train EnCodec LM ==="
pixi run python src/train.py --config configs/encodec_lm.yaml

echo "=== Step 4: Train FACodec LM ==="
pixi run python src/train.py --config configs/facodec_lm.yaml

echo "=== Step 5: Evaluate both models ==="
mkdir -p results
pixi run python scripts/run_eval.py --config configs/encodec_lm.yaml --split test --out_json results/encodec_test.json
pixi run python scripts/run_eval.py --config configs/facodec_lm.yaml --split test --out_json results/facodec_test.json

echo "=== Done! ==="
```

### 方案 B: 后台运行（推荐）

```bash
# 使用 nohup 后台运行
nohup bash run_all.sh > training.log 2>&1 &

# 查看进度
tail -f training.log

# 或使用 screen
screen -S training
bash run_all.sh
# Ctrl+A, D 分离
# screen -r training 重新连接
```

---

## 六、预期时间估算

| 步骤 | 时间 |
|------|------|
| EnCodec token 提取 (val+test) | ~10 分钟 |
| FACodec token 提取 (all) | ~30 分钟 |
| EnCodec LM 训练 (100 epochs) | ~4-6 小时 |
| FACodec LM 训练 (100 epochs) | ~4-6 小时 |
| 评估 (两个模型) | ~10 分钟 |
| **总计** | **~9-13 小时** |

---

## 七、常见问题

### Q1: CUDA out of memory
```bash
# 减小 batch_size
# 编辑 configs/encodec_lm.yaml
batch_size: 16  # 原来是 32
```

### Q2: 训练中断后恢复
```bash
# 目前不支持自动恢复，需从头训练
# 可以减少 epochs 先跑完流程
epochs: 30  # 原来是 100
```

### Q3: 验证模型是否正常学习
```bash
# 查看训练 loss 是否下降
# 初始 loss ~6.9，应该持续下降到 ~5 以下
tail -f training.log | grep "val_loss"
```

---

## 八、输出文件位置

```
checkpoints/
├── encodec_lm/
│   ├── best.pt          # 最佳验证 loss 模型
│   ├── epoch_0010.pt    # 周期性检查点
│   ├── epoch_0020.pt
│   └── tb/              # TensorBoard 日志
└── facodec_lm/
    ├── best.pt
    ├── epoch_*.pt
    └── tb/

results/
├── encodec_test.json    # EnCodec 评估结果
└── facodec_test.json    # FACodec 评估结果
```

---

## 九、完成后需要做的事

1. **提交结果**: 更新 README.md，提交最终代码和结果
2. **合并分支**: 将 `feature/speech-prediction` 合并到 main
3. **撰写报告**: 根据结果撰写项目报告

```bash
# 完成后提交
git add .
git commit -m "feat: complete training and evaluation"
git push origin feature/speech-prediction
```
