"""Training script for CodecLM.

Usage:
    pixi run python src/train.py --config configs/encodec_lm.yaml
"""

from __future__ import annotations
import argparse
import os
import sys
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from dataset import TokenDataset, collate_fn
from model import CodecLM

PAD_ID = 1025


def build_lr_schedule(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup + cosine decay."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(cfg: dict):
    device = torch.device(cfg.get("device", "cuda"))
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir / "tb")

    # ── Data ──────────────────────────────────────────────────────────────
    cache_base = cfg["token_cache_dir"]
    train_ds = TokenDataset(f"{cache_base}/train", max_len=cfg.get("max_len", 300))
    val_ds   = TokenDataset(f"{cache_base}/val",   max_len=cfg.get("max_len", 300))

    train_dl = DataLoader(
        train_ds, batch_size=cfg.get("batch_size", 32),
        shuffle=True, num_workers=4, collate_fn=collate_fn, pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=cfg.get("batch_size", 32),
        shuffle=False, num_workers=4, collate_fn=collate_fn, pin_memory=True,
    )

    # ── Model ──────────────────────────────────────────────────────────────
    model_cfg = cfg.get("model", {})
    model = CodecLM(
        vocab_size=model_cfg.get("vocab_size", 1026),
        d_model=model_cfg.get("d_model", 512),
        n_heads=model_cfg.get("n_heads", 8),
        n_layers=model_cfg.get("n_layers", 6),
        ffn_dim=model_cfg.get("ffn_dim", 2048),
        max_len=model_cfg.get("max_len", 512),
        dropout=model_cfg.get("dropout", 0.1),
    ).to(device)

    # ── Optimiser ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get("lr", 1e-4),
        weight_decay=cfg.get("weight_decay", 0.01),
    )
    epochs = cfg.get("epochs", 100)
    total_steps = epochs * len(train_dl)
    scheduler = build_lr_schedule(optimizer, cfg.get("warmup_steps", 500), total_steps)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=0.1)

    global_step = 0
    best_val_loss = float("inf")
    start_epoch = 1

    # ── Resume from checkpoint ─────────────────────────────────────────────
    resume_ckpt = cfg.get("resume", None)
    if resume_ckpt and Path(resume_ckpt).exists():
        print(f"Resuming from {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", (start_epoch - 1) * len(train_dl))
        print(f"Resumed from epoch {start_epoch - 1}, starting at epoch {start_epoch}")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_dl, desc=f"Epoch {epoch}", leave=False):
            input_ids = batch["input_ids"].to(device)   # [B, T]
            labels    = batch["labels"].to(device)       # [B, T]

            logits = model(input_ids)                    # [B, T, vocab]
            loss = criterion(
                logits.view(-1, logits.shape[-1]),
                labels.view(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            global_step += 1

            if global_step % 100 == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_dl:
                input_ids = batch["input_ids"].to(device)
                labels    = batch["labels"].to(device)
                logits = model(input_ids)
                val_loss = criterion(logits.view(-1, logits.shape[-1]), labels.view(-1))
                val_losses.append(val_loss.item())
        avg_val = sum(val_losses) / len(val_losses)
        writer.add_scalar("val/loss", avg_val, epoch)
        print(f"Epoch {epoch:3d} | train_loss={running_loss/len(train_dl):.4f} | val_loss={avg_val:.4f}")

        # ── Checkpoint ────────────────────────────────────────────────────
        ckpt = {"epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "global_step": global_step,
                "val_loss": avg_val, "best_val_loss": best_val_loss}
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            ckpt["best_val_loss"] = best_val_loss
            torch.save(ckpt, out_dir / "best.pt")
        if epoch % cfg.get("save_freq", 10) == 0:
            torch.save(ckpt, out_dir / f"epoch_{epoch:04d}.pt")

    writer.close()
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    train(cfg)
