"""Content and co-occurrence baselines (non-LLM, zero API cost).

- ``content_ranking``: user vector = recency-weighted sum of the MiniLM embeddings of the
  history items; candidates ranked by inner product (embeddings are L2-normalised, so this
  is cosine similarity up to the user-vector norm).
- ``ItemKNN``: item-item cosine similarity from train-split user co-occurrence; a candidate's
  score is the recency-weighted sum of its similarity to each history item.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def recency_weights(n: int, decay: float) -> np.ndarray:
    """w_k = exp(-decay * (n - 1 - k)): the most recent item gets weight 1."""
    return np.exp(-decay * (n - 1 - np.arange(n)))


def _rank_by_scores(candidate_ids: list[str], scores: np.ndarray) -> list[str]:
    return [iid for _, iid in sorted(zip(-scores, candidate_ids), key=lambda t: (t[0], t[1]))]


def content_ranking(
    history_ids: list[str],
    candidate_ids: list[str],
    emb: np.ndarray,
    emb_idx: dict[str, int],
    decay: float = 0.0,
) -> list[str]:
    hist = [emb_idx[i] for i in history_ids if i in emb_idx]
    if not hist:
        return sorted(candidate_ids)
    w = recency_weights(len(hist), decay)
    user = (emb[hist] * w[:, None]).sum(axis=0)
    scores = np.array(
        [float(emb[emb_idx[i]] @ user) if i in emb_idx else -np.inf for i in candidate_ids]
    )
    return _rank_by_scores(candidate_ids, scores)


class ItemKNN:
    """Cosine item-item similarity from a binary user x item matrix of train interactions."""

    def __init__(self, user_items: list[list[str]]):
        vocab = sorted({i for items in user_items for i in items})
        self.idx = {iid: k for k, iid in enumerate(vocab)}
        rows, cols = [], []
        for u, items in enumerate(user_items):
            for k in {self.idx[i] for i in items}:
                rows.append(u)
                cols.append(k)
        x = sp.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(len(user_items), len(vocab)),
        )
        co = (x.T @ x).tocsr()
        co.setdiag(0)
        co.eliminate_zeros()
        norm = np.sqrt(np.asarray(x.sum(axis=0)).ravel())
        norm[norm == 0] = 1.0
        inv = sp.diags(1.0 / norm)
        self.sim = (inv @ co @ inv).tocsr()

    def rank(self, history_ids: list[str], candidate_ids: list[str], decay: float = 0.0) -> list[str]:
        hist = [self.idx[i] for i in history_ids if i in self.idx]
        cand = [self.idx.get(i, -1) for i in candidate_ids]
        if not hist:
            return sorted(candidate_ids)
        w = recency_weights(len(hist), decay)
        user = np.asarray(self.sim[hist].multiply(w[:, None]).sum(axis=0)).ravel()
        scores = np.array([user[c] if c >= 0 else -np.inf for c in cand])
        return _rank_by_scores(candidate_ids, scores)
