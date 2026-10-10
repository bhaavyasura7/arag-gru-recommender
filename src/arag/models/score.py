"""GRU scoring: load a trained model and rank candidate items for a user."""
from __future__ import annotations

import torch

from arag.models.gru import load_model

OOV_IDX = 1


class GRUScorer:
    def __init__(self, model_dir: str):
        self.model_dir = str(model_dir)
        self.model, self.item2idx, self.config = load_model(model_dir)

    def item_indices(self, item_ids: list[str]) -> list[int]:
        return [self.item2idx.get(iid, OOV_IDX) for iid in item_ids]

    def user_vector(self, history_item_ids: list[str]) -> torch.Tensor:
        seq_len = self.config.get("seq_len", 50)
        idx = self.item_indices(history_item_ids)[-seq_len:]
        x = torch.tensor([idx], dtype=torch.long)
        with torch.no_grad():
            return self.model.encode(x)[0]

    def score_items(self, history_item_ids: list[str], candidate_item_ids: list[str]) -> dict[str, float]:
        h_u = self.user_vector(history_item_ids)
        idx = self.item_indices(candidate_item_ids)
        x = torch.tensor(idx, dtype=torch.long)
        with torch.no_grad():
            e = self.model.embedding(x)  # (N, emb)
            scores = h_u @ e.T  # (N,)
        return {iid: float(s) for iid, s in zip(candidate_item_ids, scores)}

    def rank(self, history_item_ids: list[str], candidate_item_ids: list[str]) -> list[str]:
        scores = self.score_items(history_item_ids, candidate_item_ids)
        return sorted(candidate_item_ids, key=lambda iid: (-scores[iid], iid))
