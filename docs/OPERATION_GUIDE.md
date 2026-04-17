# MDS5122 语音预测项目 - 操作指南

## 剩余工作概览

| 任务 | 状态 | 说明 |
|------|------|------|
| Task 9: End-to-End Run | ⏳ 需要手动执行 | 运行训练和评估（约需 6-12 小时） |
| Task 10: README | ⏳ 待编写 | 可在训练期间完成 |

---

## 环境说明

**重要**: 所有 `pixi run` 命令必须加 `env -u HTTP_PROXY -u HTTPS_PROXY` 前缀。
大写代理变量（端口 15732）会导致 pixi/uv 无法访问 PyPI，小写代理（端口 7897）正常工作。

为方便使用，可设置别名：
```bash
PIXI="env -u HTTP_PROXY -u HTTPS_PROXY /home/franka/.pixi/bin/pixi run"
```

---

## 一、环境准备

```bash
cd /home/franka/Development/mds5122

# 确保环境已安装（已完成）
env -u HTTP_PROXY -u HTTPS_PROXY /home/franka/.pixi/bin/pixi install

# 验证 CUDA
env -u HTTP_PROXY -u HTTPS_PROXY /home/franka/.pixi/bin/pixi run python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
# 预期输出: PyTorch: 2.3.x, CUDA: True
```

---

## 二、Token 提取（必须先完成）

### 2.1 EnCodec Token 提取

```bash
# 提取所有 split（约 15 分钟）
env -u HTTP_PROXY -u HTTPS_PROXY /home/franka/.pixi/bin/pixi run python scripts/extract_tokens.py \
    --codec encodec \
    --wsj0_root /home/franka/Development/mds5122/dataset/wsj0 \
    --out_dir /home/franka/Development/mds5122/token_cache \
    --split all --device cuda
```

**验证**:
```bash
ls token_cache/encodec/train/*.pt | wc -l  # 应为 12776
ls token_cache/encodec/val/*.pt | wc -l    # 应为 1206
ls token_cache/encodec/test/*.pt | wc -l   # 应为 651
```

### 2.2 FACodec Token 提取

```bash
# 提取所有 split（约 15 分钟）
env -u HTTP_PROXY -u HTTPS_PROXY /home/franka/.pixi/bin/pixi run python scripts/extract_tokens.py \
    --codec facodec \
    --wsj0_root /home/franka/Development/mds5122/dataset/wsj0 \
    --out_dir /home/franka/Development/mds5122/token_cache \
    --split all --device cuda \
    --facodec_repo /home/franka/Development/mds5122/FAcodec \
    --facodec_ckpt /home/franka/Development/mds5122/FAcodec/checkpoints/FAcodec.pth \
    --facodec_cfg /home/franka/Development/mds5122/FAcodec/configs/config.yml
```

**验证**:
```bash
ls token_cache/facodec/train/*.pt | wc -l  # 应为 12776
ls token_cache/facodec/val/*.pt | wc -l    # 应为 1206
ls token_cache/facodec/test/*.pt | wc -l   # 应为 651
```

---

## 三、模型训练

### 3.1 EnCodec LM 训练

```bash
# 训练 EnCodec 语言模型（约 30-40 分钟）
env -u HTTP_PROXY -u HTTPS_PROXY /home/franka/.pixi/bin/pixi run python src/train.py \
    --config configs/encodec_lm.yaml
```

**监控训练**:
```bash
# 另开终端查看 TensorBoard
env -u HTTP_PROXY -u HTTPS_PROXY /home/franka/.pixi/bin/pixi run tensorboard \
    --logdir checkpoints/encodec_lm/tb --port 6006

# 浏览器访问: http://localhost:6006
```

**预期结果**:
- 初始 val_loss ~4.0，应持续下降
- 最佳模型保存在: `checkpoints/encodec_lm/best.pt`

### 3.2 FACodec LM 训练

```bash
# 训练 FACodec 语言模型（约 30-40 分钟）
env -u HTTP_PROXY -u HTTPS_PROXY /home/franka/.pixi/bin/pixi run python src/train.py \
    --config configs/facodec_lm.yaml
```

**监控训练**:
```bash
env -u HTTP_PROXY -u HTTPS_PROXY /home/franka/.pixi/bin/pixi run tensorboard \
    --logdir checkpoints/facodec_lm/tb --port 6007
```

---

## 四、模型评估

### 4.1 评估 EnCodec 模型

```bash
mkdir -p results

env -u HTTP_PROXY -u HTTPS_PROXY /home/franka/.pixi/bin/pixi run python scripts/run_eval.py \
    --config configs/encodec_lm.yaml \
    --split test \
    --out_json results/encodec_test.json
```

### 4.2 评估 FACodec 模型

```bash
env -u HTTP_PROXY -u HTTPS_PROXY /home/franka/.pixi/bin/pixi run python scripts/run_eval.py \
    --config configs/facodec_lm.yaml \
    --split test \
    --out_json results/facodec_test.json
```

### 4.3 比较结果

```bash
env -u HTTP_PROXY -u HTTPS_PROXY /home/franka/.pixi/bin/pixi run python -c "
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
PIXI="env -u HTTP_PROXY -u HTTPS_PROXY /home/franka/.pixi/bin/pixi run"

echo "=== Step 1: Extract EnCodec tokens ==="
$PIXI python scripts/extract_tokens.py \
    --codec encodec \
    --wsj0_root /home/franka/Development/mds5122/dataset/wsj0 \
    --split all --device cuda

echo "=== Step 2: Extract FACodec tokens ==="
$PIXI python scripts/extract_tokens.py \
    --codec facodec \
    --wsj0_root /home/franka/Development/mds5122/dataset/wsj0 \
    --split all --device cuda \
    --facodec_repo /home/franka/Development/mds5122/FAcodec \
    --facodec_ckpt /home/franka/Development/mds5122/FAcodec/checkpoints/FAcodec.pth \
    --facodec_cfg /home/franka/Development/mds5122/FAcodec/configs/config.yml

echo "=== Step 3: Train EnCodec LM ==="
$PIXI python src/train.py --config configs/encodec_lm.yaml

echo "=== Step 4: Train FACodec LM ==="
$PIXI python src/train.py --config configs/facodec_lm.yaml

echo "=== Step 5: Evaluate both models ==="
mkdir -p results
$PIXI python scripts/run_eval.py --config configs/encodec_lm.yaml --split test --out_json results/encodec_test.json
$PIXI python scripts/run_eval.py --config configs/facodec_lm.yaml --split test --out_json results/facodec_test.json

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
| EnCodec token 提取 (all splits) | ~15 分钟 |
| FACodec token 提取 (all splits) | ~15 分钟 |
| EnCodec LM 训练 (100 epochs) | ~30-40 分钟 |
| FACodec LM 训练 (100 epochs) | ~30-40 分钟 |
| 评估 (两个模型) | ~10 分钟 |
| **总计** | **~1.5-2 小时** |

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
# 支持从最近的 checkpoint 自动恢复，直接重新运行训练命令即可：
env -u HTTP_PROXY -u HTTPS_PROXY /home/franka/.pixi/bin/pixi run python src/train.py \
    --config configs/encodec_lm.yaml
# 程序会自动检测 checkpoints/ 目录下的最新 checkpoint 并续训
```

### Q3: 验证模型是否正常学习
```bash
# 查看 val_loss 是否下降
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
