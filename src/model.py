"""Decoder-only Transformer for autoregressive speech token prediction.

Architecture: GPT-2 style (pre-LayerNorm, causal self-attention).

Vocabulary: 0..1023 = codec tokens, 1024 = BOS / PAD.
Total vocab_size = 1025.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_len: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        # causal mask — True means "ignore"
        mask = torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        attn = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True,
        )
        out = attn.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.proj(out))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int,
                 max_len: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class CodecLM(nn.Module):
    """Decoder-only Transformer for next-speech-token prediction.

    Args:
        vocab_size: size of token vocabulary including BOS (default 1025)
        d_model: embedding / hidden dimension
        n_heads: number of attention heads
        n_layers: number of TransformerBlock layers
        ffn_dim: feed-forward inner dimension
        max_len: maximum sequence length
        dropout: dropout probability
    """

    def __init__(
        self,
        vocab_size: int = 1025,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        ffn_dim: int = 2048,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_dim, max_len, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.tok_emb.weight
        self.max_len = max_len
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [B, T] int64
        Returns:
            logits: [B, T, vocab_size]
        """
        B, T = input_ids.shape
        assert T <= self.max_len, f"Sequence {T} exceeds max_len {self.max_len}"
        pos = torch.arange(T, device=input_ids.device)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)  # [B, T, vocab_size]

    @torch.no_grad()
    def generate(
        self,
        context: torch.Tensor,
        n_steps: int,
        temperature: float = 1.0,
        top_k: int = 200,
    ) -> torch.Tensor:
        """Autoregressively generate n_steps tokens after context.

        Args:
            context: [1, T_ctx] int64 token sequence (batch=1)
            n_steps: number of new tokens to generate
            temperature: sampling temperature
            top_k: top-k sampling cutoff (0 = greedy)

        Returns:
            generated: [1, n_steps] int64 (new tokens only)
        """
        self.eval()
        seq = context.clone()
        generated = []
        for _ in range(n_steps):
            # Trim to max_len if needed
            seq_in = seq[:, -self.max_len :]
            logits = self.forward(seq_in)[:, -1, :]  # [1, vocab_size]
            # Mask BOS/PAD from generation
            logits[:, 1024] = float("-inf")
            logits = logits / max(temperature, 1e-9)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)  # [1, 1]
            seq = torch.cat([seq, next_tok], dim=1)
            generated.append(next_tok)
        return torch.cat(generated, dim=1)  # [1, n_steps]
