"""Ranking metrics: Hit@K, NDCG@K, MRR."""
from __future__ import annotations

import math


def rank_of(target: str, ranking: list[str]) -> int:
    try:
        return ranking.index(target) + 1
    except ValueError:
        return len(ranking) + 1


def hit_at_k(rank: int, k: int) -> float:
    return 1.0 if rank <= k else 0.0


def ndcg_at_k(rank: int, k: int) -> float:
    return 1.0 / math.log2(rank + 1) if rank <= k else 0.0


def mrr(rank: int) -> float:
    return 1.0 / rank
