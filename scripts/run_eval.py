#!/usr/bin/env python
"""Full evaluation pipeline: run inference on si_et_05 and report mean metrics.

Usage:
    pixi run python scripts/run_eval.py --config configs/encodec_lm.yaml
    pixi run python scripts/run_eval.py --config configs/encodec_lm.yaml --baseline
"""

from __future__ import annotations
import argparse
import sys
import json
from pathlib import Path

import torch
import torchaudio
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from codec_wrapper import CodecWrapper
from model import CodecLM
from inference import load_model, predict_file
from evaluate import compute_metrics


def codec_baseline_file(wav_path: str, codec: CodecWrapper, device: str):
    """Encode → decode the second half of an utterance (no LM).
    Returns (reconstructed_wav, reference_wav) for computing codec ceiling."""
    wav, sr = torchaudio.load(wav_path)
    wav = wav.unsqueeze(0)                       # [1, 1, T]
    wav_24 = torchaudio.functional.resample(wav.squeeze(0), sr, 24000).unsqueeze(0)
    T_total = wav_24.shape[-1]
    ref_wav = wav_24[:, :, T_total // 2:]        # [1, 1, T_half]

    tokens = codec.encode(ref_wav, src_sr=24000) # [1, T_frames]
    recon  = codec.decode(tokens)                # [1, 1, T_recon]

    T = min(ref_wav.shape[-1], recon.shape[-1])
    return recon[:, :, :T].cpu(), ref_wav[:, :, :T].cpu()


def summarise(results: list[dict], codec_type: str, split: str, label: str):
    stoi_vals   = [r["stoi"]   for r in results]
    pesq_vals   = [r["pesq"]   for r in results]
    dnsmos_vals = [r["dnsmos"] for r in results]
    print(f"\n{'='*40}")
    print(f"Codec: {codec_type}  [{label}]")
    print(f"Split: {split}  ({len(results)} files)")
    print(f"STOI:   {sum(stoi_vals)/len(stoi_vals):.4f}")
    print(f"PESQ:   {sum(pesq_vals)/len(pesq_vals):.4f}")
    print(f"DNSMOS: {sum(dnsmos_vals)/len(dnsmos_vals):.4f}")
    print(f"{'='*40}")
    return {"stoi": sum(stoi_vals)/len(stoi_vals),
            "pesq": sum(pesq_vals)/len(pesq_vals),
            "dnsmos": sum(dnsmos_vals)/len(dnsmos_vals)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   required=True)
    parser.add_argument("--split",    default="test", choices=["test", "val"])
    parser.add_argument("--out_json", default=None)
    parser.add_argument("--baseline", action="store_true",
                        help="Also compute codec encode→decode baseline (no LM)")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device      = cfg.get("device", "cuda")
    train_max_len = cfg.get("max_len", 300)

    split_dir_map = {"test": "si_et_05", "val": "si_dt_05"}
    wav_dir  = Path(cfg["wsj0_root"]) / split_dir_map[args.split]
    wav_paths = sorted(wav_dir.rglob("*.wav"))

    # Load codec
    codec_type   = cfg["codec"]
    codec_kwargs = {}
    if codec_type == "facodec":
        codec_kwargs = {
            "facodec_repo": cfg["facodec_repo"],
            "facodec_ckpt": cfg["facodec_ckpt"],
            "facodec_cfg":  cfg["facodec_cfg"],
        }
    codec = CodecWrapper(codec_type, device=device, **codec_kwargs)

    # ── Codec baseline ────────────────────────────────────────────────────
    baseline_summary = None
    if args.baseline:
        baseline_results = []
        for wav_path in tqdm(wav_paths, desc="Codec baseline"):
            try:
                recon, ref = codec_baseline_file(str(wav_path), codec, device)
                metrics = compute_metrics(recon, ref, sr=24000)
                metrics["file"] = wav_path.stem
                baseline_results.append(metrics)
            except Exception as e:
                print(f"  WARN baseline {wav_path.name}: {e}")
        baseline_summary = summarise(baseline_results, codec_type, args.split,
                                     label="codec encode→decode, no LM")

    # ── LM prediction ─────────────────────────────────────────────────────
    ckpt_path = str(Path(cfg["out_dir"]) / "best.pt")
    lm = load_model(ckpt_path, cfg.get("model", {}), device=device)

    lm_results = []
    for wav_path in tqdm(wav_paths, desc="LM prediction"):
        try:
            pred, ref = predict_file(
                str(wav_path), codec, lm, device=device,
                temperature=cfg.get("temperature", 1.0),
                top_k=cfg.get("top_k", 200),
                train_max_len=train_max_len,
            )
            metrics = compute_metrics(pred, ref, sr=24000)
            metrics["file"] = wav_path.stem
            lm_results.append(metrics)
        except Exception as e:
            print(f"  WARN {wav_path.name}: {e}")

    lm_summary = summarise(lm_results, codec_type, args.split,
                           label="LM autoregressive prediction")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        out = {"lm": {"summary": lm_summary, "per_file": lm_results}}
        if baseline_summary is not None:
            out["baseline"] = {"summary": baseline_summary,
                               "per_file": baseline_results}
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Results saved to {args.out_json}")


if __name__ == "__main__":
    main()
