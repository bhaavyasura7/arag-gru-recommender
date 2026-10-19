"""SASRec (Kang & McAuley, 2018): self-attentive sequential recommendation.

Same interface and vocabulary as ``GRU4Rec`` so the two are compared like-for-like:
- index 0 = padding, 1 = OOV, real items from 2; sequences are right-padded
- item embedding (+ learned position embedding) -> causal Transformer blocks -> user vector
- tied output layer, full-softmax next-item loss on every prefix (as the GRU)
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class SASRec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 64,
        n_blocks: int = 2,
        n_heads: int = 2,
        dropout: float = 0.2,
        max_len: int = 50,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos = nn.Embedding(max_len, emb_dim)
        self.emb_dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_heads, dim_feedforward=emb_dim, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_blocks, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(emb_dim)
        # small init, as in the original SASRec: with N(0, 1) embeddings the tied output
        # logits start huge (loss ~30) and training stalls near random
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        with torch.no_grad():
            self.embedding.weight[0].zero_()

    def item_emb(self, idx: torch.Tensor) -> torch.Tensor:
        return self.embedding(idx)

    def all_item_emb(self) -> torch.Tensor:
        return self.embedding.weight

    def step_outputs(self, x: torch.Tensor) -> torch.Tensor:
        """Per-step outputs (B, T, emb). Position t only attends to positions <= t (causal),
        so right-padding never leaks into a real position."""
        t = x.shape[1]
        positions = torch.arange(t, device=x.device)
        h = self.embedding(x) + self.pos(positions)[None]
        h = self.emb_dropout(h) * (x != 0).unsqueeze(-1)
        causal = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=1)
        h = self.blocks(h, mask=causal)
        return self.norm(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(self.step_outputs(x), self.all_item_emb())

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """User vector = output at the last real (non-pad) position of each right-padded row."""
        out = self.step_outputs(x)
        last = ((x != 0).sum(dim=1) - 1).clamp(min=0)
        return out[torch.arange(x.shape[0], device=x.device), last]


def save_model(directory, model: SASRec, item2idx: dict[str, int], config: dict) -> None:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), d / "model.pt")
    with open(d / "item_map.json", "w", encoding="utf-8") as fh:
        json.dump({"item2idx": item2idx}, fh)
    with open(d / "config.json", "w", encoding="utf-8") as fh:
        json.dump(config, fh)


def load_model(directory) -> tuple[SASRec, dict[str, int], dict]:
    d = Path(directory)
    with open(d / "config.json", "r", encoding="utf-8") as fh:
        config = json.load(fh)
    with open(d / "item_map.json", "r", encoding="utf-8") as fh:
        item2idx = json.load(fh)["item2idx"]
    model = SASRec(
        vocab_size=len(item2idx) + 2,
        emb_dim=config["emb_dim"],
        n_blocks=config["n_blocks"],
        n_heads=config["n_heads"],
        dropout=config["dropout"],
        max_len=config["seq_len"],
    )
    model.load_state_dict(torch.load(d / "model.pt", map_location="cpu"))
    model.eval()
    return model, item2idx, config
