#!/usr/bin/env python
"""Full evaluation pipeline: run inference on si_et_05 and report mean metrics.

Usage:
    pixi run python scripts/run_eval.py --config configs/encodec_lm.yaml
"""

from __future__ import annotations
import argparse
import sys
import json
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from codec_wrapper import CodecWrapper
from model import CodecLM
from inference import load_model, predict_file
from evaluate import compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--out_json", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = cfg.get("device", "cuda")
    split_dir_map = {"test": "si_et_05", "val": "si_dt_05"}
    wav_dir = Path(cfg["wsj0_root"]) / split_dir_map[args.split]
    wav_paths = sorted(wav_dir.rglob("*.wav"))

    # Load codec
    codec_type = cfg["codec"]
    codec_kwargs = {}
    if codec_type == "facodec":
        codec_kwargs = {
            "facodec_repo": cfg["facodec_repo"],
            "facodec_ckpt": cfg["facodec_ckpt"],
            "facodec_cfg":  cfg["facodec_cfg"],
        }
    codec = CodecWrapper(codec_type, device=device, **codec_kwargs)

    # Load LM
    ckpt_path = str(Path(cfg["out_dir"]) / "best.pt")
    lm = load_model(ckpt_path, cfg.get("model", {}), device=device)

    all_results = []
    for wav_path in tqdm(wav_paths, desc="Evaluating"):
        try:
            pred, ref = predict_file(
                str(wav_path), codec, lm, device=device,
                temperature=cfg.get("temperature", 1.0),
                top_k=cfg.get("top_k", 200),
            )
            metrics = compute_metrics(pred, ref, sr=24000)
            metrics["file"] = wav_path.stem
            all_results.append(metrics)
        except Exception as e:
            print(f"  WARN: {wav_path.name}: {e}")

    stoi_vals  = [r["stoi"]  for r in all_results]
    pesq_vals  = [r["pesq"]  for r in all_results]
    dnsmos_vals= [r["dnsmos"] for r in all_results]

    print(f"\n{'='*40}")
    print(f"Codec: {codec_type}")
    print(f"Split: {args.split}  ({len(all_results)} files)")
    print(f"STOI:   {sum(stoi_vals)/len(stoi_vals):.4f}")
    print(f"PESQ:   {sum(pesq_vals)/len(pesq_vals):.4f}")
    print(f"DNSMOS: {sum(dnsmos_vals)/len(dnsmos_vals):.4f}")
    print(f"{'='*40}")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({"summary": {"stoi": sum(stoi_vals)/len(stoi_vals),
                                   "pesq": sum(pesq_vals)/len(pesq_vals),
                                   "dnsmos": sum(dnsmos_vals)/len(dnsmos_vals)},
                       "per_file": all_results}, f, indent=2)
        print(f"Results saved to {args.out_json}")


if __name__ == "__main__":
    main()
