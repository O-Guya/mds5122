"""Autoregressive speech prediction inference.

For each test utterance:
1. Load wav, resample to 24kHz, split at midpoint in samples.
2. Encode context half → token sequence.
3. Feed context tokens to CodecLM.generate() → predicted future tokens.
4. Decode predicted tokens → future waveform.
5. Return: (predicted_wav, reference_future_wav, sample_rate=24000)

Both wavs are returned at 24 kHz for evaluation.
"""

from __future__ import annotations
import sys
from pathlib import Path
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).parent))
from codec_wrapper import CodecWrapper, BOS_ID
from model import CodecLM


def load_model(ckpt_path: str, model_cfg: dict, device: str = "cuda") -> CodecLM:
    model = CodecLM(
        vocab_size=model_cfg.get("vocab_size", 1025),
        d_model=model_cfg.get("d_model", 512),
        n_heads=model_cfg.get("n_heads", 8),
        n_layers=model_cfg.get("n_layers", 6),
        ffn_dim=model_cfg.get("ffn_dim", 2048),
        max_len=model_cfg.get("max_len", 512),
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


@torch.no_grad()
def predict_file(
    wav_path: str,
    codec: CodecWrapper,
    lm: CodecLM,
    device: str = "cuda",
    temperature: float = 1.0,
    top_k: int = 200,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict future speech given the first half of an utterance.

    Returns:
        pred_wav: [1, 1, T_future] float32 at 24 kHz
        ref_wav:  [1, 1, T_future] float32 at 24 kHz  (ground truth second half)
    """
    wav, sr = torchaudio.load(wav_path)         # [1, T_orig]
    wav = wav.unsqueeze(0)                       # [1, 1, T_orig]

    # Resample once to 24 kHz for reference waveform
    wav_24 = torchaudio.functional.resample(wav.squeeze(0), sr, 24000).unsqueeze(0)
    T_total = wav_24.shape[-1]
    mid = T_total // 2

    context_wav = wav_24[:, :, :mid]            # [1, 1, mid]
    ref_wav     = wav_24[:, :, mid:]            # [1, 1, T_total-mid]

    # Encode context at original sr (codec_wrapper resamples internally)
    # Here wav is already 24kHz, so src_sr=24000
    ctx_tokens = codec.encode(context_wav, src_sr=24000)  # [1, T_ctx]
    n_future = ref_wav.shape[-1] // (24000 // codec.frame_rate)  # approx token count

    # Prepend BOS and generate
    bos = torch.tensor([[BOS_ID]], dtype=torch.long, device=device)
    ctx_ids = torch.cat([bos, ctx_tokens.to(device)], dim=1)

    pred_tokens = lm.generate(ctx_ids, n_steps=n_future,
                              temperature=temperature, top_k=top_k)  # [1, n_future]

    pred_wav = codec.decode(pred_tokens)        # [1, 1, T_pred]
    # Align length to reference
    T_ref = ref_wav.shape[-1]
    T_pred = pred_wav.shape[-1]
    if T_pred > T_ref:
        pred_wav = pred_wav[:, :, :T_ref]
    elif T_pred < T_ref:
        pad = torch.zeros(1, 1, T_ref - T_pred)
        pred_wav = torch.cat([pred_wav.cpu(), pad], dim=-1)

    return pred_wav.cpu(), ref_wav.cpu()
