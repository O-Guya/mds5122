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
        vocab_size=model_cfg.get("vocab_size", 1026),
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
    train_max_len: int = 300,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict future speech given the first half of an utterance.

    Args:
        train_max_len: max sequence length used during training. Context and
            generation are both capped to train_max_len // 2 so that inference
            stays within the positional-embedding range seen during training.

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

    # Encode context
    ctx_tokens = codec.encode(context_wav, src_sr=24000)  # [1, T_ctx]

    # Cap context and generation to training distribution:
    # BOS(1) + ctx(n) + future(n) <= train_max_len  =>  n <= (train_max_len - 1) // 2
    max_ctx = (train_max_len - 1) // 2          # 149 for train_max_len=300
    max_future = train_max_len - 1 - max_ctx    # 150 for train_max_len=300
    if ctx_tokens.shape[1] > max_ctx:
        ctx_tokens = ctx_tokens[:, -max_ctx:]   # keep most-recent context tokens

    n_future = ref_wav.shape[-1] // (24000 // codec.frame_rate)
    n_future = min(n_future, max_future)

    # Prepend BOS and generate
    bos = torch.tensor([[BOS_ID]], dtype=torch.long, device=device)
    ctx_ids = torch.cat([bos, ctx_tokens.to(device)], dim=1)

    pred_tokens = lm.generate(ctx_ids, n_steps=n_future,
                              temperature=temperature, top_k=top_k)  # [1, n_future]

    pred_wav = codec.decode(pred_tokens)        # [1, 1, T_pred]
    # Align to the shorter of the two — never pad prediction with silence,
    # as that would artificially deflate STOI/PESQ for long utterances.
    T_ref = ref_wav.shape[-1]
    T_pred = pred_wav.shape[-1]
    T = min(T_ref, T_pred)
    pred_wav = pred_wav[:, :, :T].cpu()
    ref_wav  = ref_wav[:, :, :T].cpu()

    return pred_wav, ref_wav
