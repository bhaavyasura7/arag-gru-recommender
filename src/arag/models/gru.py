"""GRU4Rec-style sequential model with tied input/output embeddings.

- item embedding 64-d -> 1-layer GRU (hidden) -> dropout -> user vector
- vocab: index 0 = padding, 1 = OOV, real items start at 2
- full-softmax next-item prediction over the vocab, trained on every prefix
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class GRU4Rec(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 64, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden = hidden
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden, emb_dim, bias=False) if hidden != emb_dim else nn.Identity()

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Training forward: per-step logits (B, T, vocab)."""
        emb = self.embedding(x)
        emb = self.dropout(emb)
        lengths_cpu = lengths.to("cpu")
        packed = pack_padded_sequence(emb, lengths_cpu, batch_first=True, enforce_sorted=False)
        output, _ = self.gru(packed)
        out_padded, _ = pad_packed_sequence(output, batch_first=True, total_length=x.shape[1])
        out_padded = self.dropout(out_padded)
        out_padded = self.proj(out_padded)  # (B, T, emb_dim)
        logits = F.linear(out_padded, self.embedding.weight)  # (B, T, vocab)
        return logits

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """User vector h_u = last valid hidden state, projected to emb_dim."""
        lengths = (x != 0).sum(dim=1)
        emb = self.embedding(x)
        packed = pack_padded_sequence(emb, lengths.to("cpu"), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)  # (1, B, hidden)
        h_u = h.squeeze(0)
        h_u = self.proj(h_u)
        return h_u

    def score_items(self, h_u: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        """Scores h_u . e_i for each item index i."""
        e = self.embedding(item_idx)  # (B, emb) or (N, emb)
        if e.dim() == 2 and h_u.dim() == 2 and e.shape[0] == h_u.shape[0]:
            return (h_u * e).sum(dim=1)
        return h_u @ e.T


def compute_lengths(x: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    return (x != pad_idx).sum(dim=1)


def save_model(directory, model: GRU4Rec, item2idx: dict[str, int], config: dict) -> None:
    import json
    from pathlib import Path

    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), d / "model.pt")
    with open(d / "item_map.json", "w", encoding="utf-8") as fh:
        json.dump({"item2idx": item2idx}, fh)
    with open(d / "config.json", "w", encoding="utf-8") as fh:
        json.dump(config, fh)


def load_model(directory) -> tuple[GRU4Rec, dict[str, int], dict]:
    import json
    from pathlib import Path

    d = Path(directory)
    with open(d / "config.json", "r", encoding="utf-8") as fh:
        config = json.load(fh)
    with open(d / "item_map.json", "r", encoding="utf-8") as fh:
        item2idx = json.load(fh)["item2idx"]
    vocab_size = len(item2idx) + 2
    model = GRU4Rec(
        vocab_size=vocab_size,
        emb_dim=config.get("emb_dim", 64),
        hidden=config.get("hidden", 128),
        dropout=config.get("dropout", 0.2),
    )
    model.load_state_dict(torch.load(d / "model.pt", map_location="cpu"))
    model.eval()
    return model, item2idx, config
