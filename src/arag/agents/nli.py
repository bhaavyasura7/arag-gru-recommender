"""NLI agent: judge each candidate as support/neutral/contradict with a score."""
from __future__ import annotations

from dataclasses import dataclass, field

from arag.agents.prompts import format_candidates, load_prompt, render
from arag.config import Config
from arag.llm.client import LLMClient
from arag.llm.schemas import NLIOutput, to_strict_schema


@dataclass
class NLIResult:
    items: dict[str, dict] = field(default_factory=dict)
    nli_missing: int = 0
    reasked: bool = False
    fallback_calls: int = 0

    def accepted(self, theta: int) -> list[tuple[str, dict]]:
        acc = [(iid, d) for iid, d in self.items.items() if d["score"] >= theta]
        acc.sort(key=lambda kv: -kv[1]["score"])
        return acc

    def scores(self) -> dict[str, int]:
        return {iid: d["score"] for iid, d in self.items.items()}


def select_accepted(nli: NLIResult, theta: int) -> tuple[list[tuple[str, dict]], bool]:
    """Accepted items ordered by score; if none pass theta, use top-3 by score.

    Returns (accepted, empty_fallback).
    """
    accepted = nli.accepted(theta)
    if accepted:
        return accepted, False
    top = sorted(nli.items.items(), key=lambda kv: -kv[1]["score"])[:3]
    return [(iid, d) for iid, d in top], True


def _nli_call(client: LLMClient, cfg: Config, user_context: str, chunk: list[dict]):
    template = load_prompt(cfg, "nli")
    prompt = render(template, USER_CONTEXT=user_context, CANDIDATES=format_candidates(chunk))
    schema = to_strict_schema(NLIOutput, "nli")
    res = client.complete_validated("nli", prompt, schema, NLIOutput, fallback=NLIOutput(results=[]))
    return res.obj, res


def run_nli(
    client: LLMClient,
    cfg: Config,
    user_context: str,
    candidates: list[dict],
    per_call: int = 5,
) -> NLIResult:
    result = NLIResult()
    results: dict[str, dict] = result.items
    from concurrent.futures import ThreadPoolExecutor

    chunks = [candidates[i : i + per_call] for i in range(0, len(candidates), per_call)]
    if len(chunks) > 1:
        with ThreadPoolExecutor(max_workers=min(len(chunks), 4)) as ex:
            out_and_res = list(ex.map(lambda c: _nli_call(client, cfg, user_context, c), chunks))
    else:
        out_and_res = [_nli_call(client, cfg, user_context, c) for c in chunks]

    for out, res in out_and_res:
        result.fallback_calls += int(res.fallback)
        for item in out.results:
            if item.item_id not in results:
                results[item.item_id] = {
                    "label": item.label,
                    "score": item.score,
                    "rationale": item.rationale,
                }

    all_ids = [c["item_id"] for c in candidates]
    missing = [iid for iid in all_ids if iid not in results]
    reasked = bool(missing)
    if missing:
        missing_items = [c for c in candidates if c["item_id"] in missing]
        out2, res2 = _nli_call(client, cfg, user_context, missing_items)
        result.fallback_calls += int(res2.fallback)
        for item in out2.results:
            if item.item_id in results:
                continue
            results[item.item_id] = {
                "label": item.label,
                "score": item.score,
                "rationale": item.rationale,
            }

    still_missing = [iid for iid in all_ids if iid not in results]
    for iid in still_missing:
        results[iid] = {"label": "neutral", "score": 5, "rationale": "missing after re-ask"}

    result.nli_missing = len(still_missing)
    result.reasked = reasked
    return result
