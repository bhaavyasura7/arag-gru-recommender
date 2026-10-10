"""Context Summary Agent (CSA)."""
from __future__ import annotations

from arag.agents.prompts import format_accepted, load_prompt, render
from arag.config import Config
from arag.llm.client import LLMClient
from arag.llm.schemas import CSAOutput, to_strict_schema


def fallback_csa() -> CSAOutput:
    return CSAOutput(summary="")


def run_csa(
    client: LLMClient,
    cfg: Config,
    s_user: str,
    accepted: list[dict],
    empty_fallback: bool = False,
):
    """``accepted`` is a list of item dicts (item_id, score, text) ordered by score."""
    template = load_prompt(cfg, "csa")
    prompt = render(template, S_USER=s_user, ACCEPTED_ITEMS=format_accepted(accepted))
    schema = to_strict_schema(CSAOutput, "csa")
    res = client.complete_validated("csa", prompt, schema, CSAOutput, fallback=fallback_csa())
    return res.obj, res, empty_fallback
