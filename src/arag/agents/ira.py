"""Item Ranker Agent (IRA)."""
from __future__ import annotations

from arag.agents.prompts import format_candidates, load_prompt, render
from arag.config import Config
from arag.llm.client import LLMClient
from arag.llm.schemas import IRAOutput, to_strict_schema


def fallback_ira(candidate_ids: list[str]) -> IRAOutput:
    return IRAOutput(ranking=list(candidate_ids), reasons=[])


def run_ira(
    client: LLMClient,
    cfg: Config,
    s_user: str,
    s_ctx: str,
    candidates: list[dict],
):
    template = load_prompt(cfg, "ira")
    prompt = render(template, S_USER=s_user, S_CTX=s_ctx, CANDIDATES=format_candidates(candidates))
    schema = to_strict_schema(IRAOutput, "ira")
    res = client.complete_validated(
        "ira", prompt, schema, IRAOutput, fallback=fallback_ira([c["item_id"] for c in candidates])
    )
    return res.obj, res
