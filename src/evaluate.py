"""Compute STOI, PESQ, and DNSMOS on predicted vs reference speech.

TorchMetrics handles STOI and PESQ; DNSMOS uses torchmetrics.audio.dnsmos.
All metrics expect float tensors at the correct sample rate.

STOI:  fs=24000, expects [B, T] tensors at 24kHz
PESQ:  only supports 8kHz and 16kHz — we downsample to 16kHz internally
DNSMOS: non-intrusive — only needs predicted speech
"""

from __future__ import annotations
import torch
import torchaudio
from torchmetrics.audio import ShortTimeObjectiveIntelligibility, PerceptualEvaluationSpeechQuality
from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore


def compute_metrics(
    pred_wav: torch.Tensor,
    ref_wav: torch.Tensor,
    sr: int = 24000,
) -> dict[str, float]:
    """Compute STOI, PESQ, DNSMOS for one utterance pair.

    Args:
        pred_wav: [1, 1, T] or [1, T] float32 predicted waveform
        ref_wav:  [1, 1, T] or [1, T] float32 reference waveform
        sr: sample rate (default 24000)

    Returns:
        dict with keys 'stoi', 'pesq', 'dnsmos'
    """
    # Flatten to [T]
    pred = pred_wav.squeeze().float()
    ref  = ref_wav.squeeze().float()

    # Match lengths
    L = min(pred.shape[0], ref.shape[0])
    pred, ref = pred[:L], ref[:L]

    results = {}

    # ── STOI ─────────────────────────────────────────────────────────────
    stoi_metric = ShortTimeObjectiveIntelligibility(fs=sr, extended=False)
    results["stoi"] = stoi_metric(pred.unsqueeze(0), ref.unsqueeze(0)).item()

    # ── PESQ (requires 8kHz or 16kHz) ────────────────────────────────────
    sr_pesq = 16000
    pred_16 = torchaudio.functional.resample(pred, sr, sr_pesq)
    ref_16  = torchaudio.functional.resample(ref,  sr, sr_pesq)
    pesq_metric = PerceptualEvaluationSpeechQuality(fs=sr_pesq, mode="wb")
    results["pesq"] = pesq_metric(pred_16.unsqueeze(0), ref_16.unsqueeze(0)).item()

    # ── DNSMOS (non-intrusive, no reference needed) ───────────────────────
    dnsmos_metric = DeepNoiseSuppressionMeanOpinionScore(fs=sr, personalized=False)
    dnsmos_vals = dnsmos_metric(pred.unsqueeze(0))
    # DNSMOS returns [p808_mos, mos_sig, mos_bak, mos_ovr]
    results["dnsmos"] = dnsmos_vals[0].item()  # p808_mos is the overall MOS

    return results
