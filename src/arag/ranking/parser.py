"""Ranking parser: repair malformed LLM rankings into a valid permutation."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RepairStats:
    unknown: int = 0
    duplicate: int = 0
    missing: int = 0

    @property
    def total(self) -> int:
        return self.unknown + self.duplicate + self.missing


def parse_ranking(
    ranking: list[str],
    candidate_ids: list[str],
    nli_scores: dict[str, int] | None = None,
) -> tuple[list[str], RepairStats]:
    """Return (valid_20_permutation, RepairStats).

    - drop unknown ids
    - drop duplicates (keep first)
    - append missing ids (by NLI score desc, else list order)
    """
    stats = RepairStats()
    valid = set(candidate_ids)
    seen: set[str] = set()
    out: list[str] = []
    for iid in ranking:
        if iid not in valid:
            stats.unknown += 1
            continue
        if iid in seen:
            stats.duplicate += 1
            continue
        seen.add(iid)
        out.append(iid)

    missing = [iid for iid in candidate_ids if iid not in seen]
    if missing:
        stats.missing = len(missing)
        if nli_scores is not None:
            missing = sorted(
                missing, key=lambda i: (-int(nli_scores.get(i, 0)), candidate_ids.index(i))
            )
        out.extend(missing)

    return out, stats
