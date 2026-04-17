#!/usr/bin/env python
"""Pre-extract codec tokens for all WSJ0 splits and save as .pt files.

Usage:
    pixi run python scripts/extract_tokens.py \
        --codec encodec \
        --wsj0_root /home/franka/Development/mds5122/dataset/wsj0 \
        --out_dir /home/franka/Development/mds5122/token_cache \
        --split all

    pixi run python scripts/extract_tokens.py \
        --codec facodec \
        --facodec_repo /home/franka/Development/mds5122/FAcodec \
        --facodec_ckpt /path/to/facodec.pth \
        --facodec_cfg  /home/franka/Development/mds5122/FAcodec/configs/config.yml \
        --wsj0_root /home/franka/Development/mds5122/dataset/wsj0 \
        --out_dir /home/franka/Development/mds5122/token_cache \
        --split all
"""

import argparse
import sys
from pathlib import Path
import torch
import torchaudio
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from codec_wrapper import CodecWrapper

SPLIT_DIRS = {
    "train": "si_tr_s",
    "val":   "si_dt_05",
    "test":  "si_et_05",
}


def extract_split(codec: CodecWrapper, split: str, wsj0_root: Path,
                  out_root: Path, batch_size: int = 8):
    split_dir = wsj0_root / SPLIT_DIRS[split]
    wav_paths = sorted(split_dir.rglob("*.wav"))
    out_dir = out_root / codec.codec_type / split
    out_dir.mkdir(parents=True, exist_ok=True)

    already_done = {p.stem for p in out_dir.glob("*.pt")}

    print(f"[{split}] {len(wav_paths)} files -> {out_dir}")
    for path in tqdm(wav_paths, desc=split):
        stem = path.stem
        if stem in already_done:
            continue
        wav, sr = torchaudio.load(str(path))          # [1, T]
        wav = wav.unsqueeze(0)                         # [1, 1, T]
        codes = codec.encode(wav, src_sr=sr)           # [1, T_frames]
        tokens = codes[0].cpu()                        # [T_frames] int64
        torch.save(tokens, out_dir / f"{stem}.pt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--codec",        required=True, choices=["encodec", "facodec"])
    p.add_argument("--wsj0_root",    default="/home/franka/Development/mds5122/dataset/wsj0")
    p.add_argument("--out_dir",      default="/home/franka/Development/mds5122/token_cache")
    p.add_argument("--split",        default="all", choices=["train","val","test","all"])
    p.add_argument("--device",       default="cuda")
    p.add_argument("--facodec_repo", default="/home/franka/Development/mds5122/FAcodec")
    p.add_argument("--facodec_ckpt", default=None)
    p.add_argument("--facodec_cfg",  default=None)
    args = p.parse_args()

    codec = CodecWrapper(
        args.codec, device=args.device,
        facodec_repo=args.facodec_repo if args.codec == "facodec" else None,
        facodec_ckpt=args.facodec_ckpt if args.codec == "facodec" else None,
        facodec_cfg=args.facodec_cfg  if args.codec == "facodec" else None,
    )

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for s in splits:
        extract_split(codec, s, Path(args.wsj0_root), Path(args.out_dir))


if __name__ == "__main__":
    main()
