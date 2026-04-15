"""WSJ0 dataset that loads pre-extracted token sequences from a cache directory.

Cache format: one `.pt` file per utterance, containing a 1-D int64 tensor of token IDs.
"""

from __future__ import annotations
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset

BOS_ID = 1024
PAD_ID = 1024  # same as BOS for simplicity


class TokenDataset(Dataset):
    """Load pre-tokenised sequences for next-token-prediction LM training.

    Args:
        cache_dir: directory containing *.pt files of shape [T]
        max_len: maximum sequence length (tokens). Longer sequences are
                 randomly cropped; shorter ones are padded with PAD_ID.
        split: 'train' | 'val' | 'test' (informational only; caller
               points to the correct cache_dir)
    """

    def __init__(self, cache_dir: str, max_len: int = 400, split: str = "train"):
        super().__init__()
        self.paths = sorted(Path(cache_dir).glob("*.pt"))
        assert len(self.paths) > 0, f"No .pt files found in {cache_dir}"
        self.max_len = max_len
        self.split = split

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        tokens = torch.load(self.paths[idx])  # [T] int64
        T = tokens.shape[0]

        if T > self.max_len:
            start = random.randint(0, T - self.max_len)
            tokens = tokens[start : start + self.max_len]

        # Prepend BOS
        tokens = torch.cat([torch.tensor([BOS_ID], dtype=torch.long), tokens])

        # Pad to max_len+1 if needed
        pad_len = (self.max_len + 1) - tokens.shape[0]
        if pad_len > 0:
            tokens = torch.cat([tokens, torch.full((pad_len,), PAD_ID, dtype=torch.long)])

        # input = tokens[:-1], target = tokens[1:]
        return {"input_ids": tokens[:-1], "labels": tokens[1:]}


def collate_fn(batch: list[dict]) -> dict:
    input_ids = torch.stack([b["input_ids"] for b in batch])
    labels    = torch.stack([b["labels"]    for b in batch])
    return {"input_ids": input_ids, "labels": labels}
