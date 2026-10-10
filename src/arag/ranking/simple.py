"""Simple (non-LLM) baselines: random and popularity ranking."""
from __future__ import annotations

import hashlib
import random


def seed_for(seed: int, *keys: str) -> int:
    h = hashlib.md5("|".join(map(str, keys)).encode("utf-8")).digest()[:8]
    return seed + int.from_bytes(h, "big")


def random_ranking(candidate_ids: list[str], seed: int) -> list[str]:
    ids = list(candidate_ids)
    random.Random(seed).shuffle(ids)
    return ids


def popularity_ranking(candidate_ids: list[str], popularity: dict[str, int]) -> list[str]:
    return sorted(candidate_ids, key=lambda i: (-int(popularity.get(i, 0)), i))
