"""Late fusion of the LLM (IRA) ranking and the GRU score ranking."""
from __future__ import annotations


def rank_to_scores(ranking: list[str]) -> dict[str, float]:
    n = len(ranking)
    if n <= 1:
        return {iid: 0.0 for iid in ranking}
    return {iid: 1.0 - idx / (n - 1) for idx, iid in enumerate(ranking)}


def fuse(
    candidate_ids: list[str],
    rank_llm: list[str],
    rank_gru: list[str],
    alpha: float,
    nli_scores: dict[str, int] | None = None,
) -> list[str]:
    """Rank candidates by ``alpha * s_llm + (1 - alpha) * s_gru``.

    Ties broken by higher NLI score, then item_id. ``alpha == 1`` reproduces the LLM
    order exactly; ``alpha == 0`` reproduces the GRU order exactly.
    """
    s_llm = rank_to_scores(rank_llm)
    s_gru = rank_to_scores(rank_gru)

    def key(iid: str):
        final = alpha * s_llm[iid] + (1 - alpha) * s_gru[iid]
        nli = nli_scores.get(iid, 0) if nli_scores else 0
        return (-final, -nli, iid)

    return sorted(candidate_ids, key=key)
