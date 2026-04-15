"""Unified interface for EnCodec and FACodec.

Both codecs operate at 24 kHz. WSJ0 is 16 kHz — resample before encoding.

EnCodec: encode returns List[(codes [B,K,T], scale)]. We use bandwidth=1.5
kbps → K=2 codebooks. We take only codebook 0 (index 0 along dim 1).

FACodec: we use content codes with n_c=1 → codes_c of shape [B, 1, T].
FACodec requires its own encoder + quantizer model dict.
"""

from __future__ import annotations
import sys
import torch
import torchaudio
from pathlib import Path

# Codec vocab size (1024) + one BOS token
VOCAB_SIZE = 1024
BOS_ID = 1024  # also used as padding


def _resample(wav: torch.Tensor, orig_sr: int, target_sr: int = 24000) -> torch.Tensor:
    """Resample [B, 1, T] or [1, T] tensor."""
    if orig_sr == target_sr:
        return wav
    return torchaudio.functional.resample(wav, orig_sr, target_sr)


class CodecWrapper:
    """Wrap EnCodec or FACodec into a common encode/decode API.

    Args:
        codec_type: "encodec" or "facodec"
        device: torch device string
        facodec_repo: path to FAcodec repo root (needed for sys.path)
        facodec_ckpt: path to FACodec checkpoint .pth
        facodec_cfg: path to FACodec config .yml
    """

    def __init__(
        self,
        codec_type: str,
        device: str = "cuda",
        facodec_repo: str | None = None,
        facodec_ckpt: str | None = None,
        facodec_cfg: str | None = None,
    ):
        assert codec_type in ("encodec", "facodec"), f"Unknown codec: {codec_type}"
        self.codec_type = codec_type
        self.device = torch.device(device)
        self.sample_rate = 24000  # both codecs operate at 24 kHz

        if codec_type == "encodec":
            self._init_encodec()
        else:
            assert facodec_repo and facodec_ckpt and facodec_cfg, (
                "facodec_repo, facodec_ckpt, facodec_cfg are required for facodec"
            )
            self._init_facodec(facodec_repo, facodec_ckpt, facodec_cfg)

    # ------------------------------------------------------------------
    # EnCodec init
    # ------------------------------------------------------------------
    def _init_encodec(self):
        from encodec.model import EncodecModel
        model = EncodecModel.encodec_model_24khz(pretrained=True)
        model.set_target_bandwidth(1.5)  # 2 codebooks at 24kHz
        model = model.to(self.device).eval()
        self._encodec = model
        # frame rate for 24kHz with ratios [8,5,4,2]: stride = 320
        self.frame_rate = 75  # frames per second

    # ------------------------------------------------------------------
    # FACodec init
    # ------------------------------------------------------------------
    def _init_facodec(self, repo: str, ckpt: str, cfg_path: str):
        if repo not in sys.path:
            sys.path.insert(0, repo)
        import yaml
        from modules.commons import build_model, recursive_munch
        cfg = yaml.safe_load(open(cfg_path))
        model_dict = build_model(recursive_munch(cfg["model_params"]))
        ckpt_data = torch.load(ckpt, map_location="cpu")
        params = ckpt_data["net"] if "net" in ckpt_data else ckpt_data
        for key in params:
            if key in model_dict:
                model_dict[key].load_state_dict(params[key])
        for key in model_dict:
            model_dict[key] = model_dict[key].to(self.device).eval()
        self._facodec = model_dict
        # 24kHz / 300 hop = 80 fps
        self.frame_rate = 80

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode(self, wav: torch.Tensor, src_sr: int = 16000) -> torch.Tensor:
        """Encode waveform to token IDs.

        Args:
            wav: [B, 1, T_samples] float32 tensor at src_sr
            src_sr: sample rate of input wav

        Returns:
            codes: [B, T_frames] int64 — first codebook only, values 0..1023
        """
        wav = _resample(wav.to(self.device), src_sr, self.sample_rate)
        if self.codec_type == "encodec":
            return self._encode_encodec(wav)
        else:
            return self._encode_facodec(wav)

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode token IDs back to waveform.

        Args:
            codes: [B, T_frames] int64, values 0..1023

        Returns:
            wav: [B, 1, T_samples] float32 at 24 kHz
        """
        if self.codec_type == "encodec":
            return self._decode_encodec(codes)
        else:
            return self._decode_facodec(codes)

    # ------------------------------------------------------------------
    # EnCodec encode / decode
    # ------------------------------------------------------------------
    def _encode_encodec(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: [B, 1, T]
        frames = self._encodec.encode(wav)          # List[(codes [B,K,T], scale)]
        codes = frames[0][0]                        # [B, K, T_frames]
        return codes[:, 0, :]                       # [B, T_frames] first codebook

    def _decode_encodec(self, codes: torch.Tensor) -> torch.Tensor:
        # codes: [B, T_frames]
        # EnCodec decode expects List[(codes [B,K,T], scale)]
        codes_bkt = codes.unsqueeze(1)              # [B, 1, T]
        # Pad to n_q codebooks with zeros (other codebooks won't contribute much)
        n_q = self._encodec.quantizer.n_q
        if n_q > 1:
            pad = torch.zeros(
                codes.shape[0], n_q - 1, codes.shape[1],
                dtype=codes.dtype, device=codes.device
            )
            codes_bkt = torch.cat([codes_bkt, pad], dim=1)  # [B, n_q, T]
        encoded_frame = (codes_bkt, None)
        wav = self._encodec.decode([encoded_frame])         # [B, 1, T_out]
        return wav

    # ------------------------------------------------------------------
    # FACodec encode / decode
    # ------------------------------------------------------------------
    def _encode_facodec(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: [B, 1, T]  ->  encoder expects [B, 1, T]
        encoder = self._facodec["encoder"]
        quantizer = self._facodec["quantizer"]
        z = encoder(wav)
        # n_c=1: use 1 content codebook
        codes_list, _ = quantizer.encode(z, wav, n_c=1)
        codes_c = codes_list[0]                     # [B, 1, T_frames]
        return codes_c[:, 0, :]                     # [B, T_frames]

    def _decode_facodec(self, codes: torch.Tensor) -> torch.Tensor:
        # codes: [B, T_frames]
        quantizer = self._facodec["quantizer"]
        decoder = self._facodec["decoder"]
        # Reconstruct latent from content codes only (prosody/timbre=zero)
        z_c = quantizer.content_quantizer.from_codes(codes.unsqueeze(1))[0]
        wav = decoder(z_c)                          # [B, 1, T_out]
        return wav
