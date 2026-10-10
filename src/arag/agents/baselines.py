"""LLM baseline rankers: LLM-Recency and LLM-Vanilla-RAG."""
from __future__ import annotations

from arag.agents.prompts import format_candidates, load_prompt, render
from arag.config import Config
from arag.llm.client import LLMClient
from arag.llm.schemas import RankingOutput, to_strict_schema


def _run_ranking(
    client: LLMClient,
    cfg: Config,
    agent: str,
    template_name: str,
    history_lines: list[str],
    candidates: list[dict],
):
    template = load_prompt(cfg, template_name)
    prompt = render(
        template,
        HISTORY="\n".join(history_lines),
        CANDIDATES=format_candidates(candidates),
    )
    schema = to_strict_schema(RankingOutput, "ranking")
    fallback = RankingOutput(ranking=[c["item_id"] for c in candidates])
    res = client.complete_validated(agent, prompt, schema, RankingOutput, fallback=fallback)
    return res.obj, res


def run_recency(client: LLMClient, cfg: Config, history_lines: list[str], candidates: list[dict]):
    return _run_ranking(client, cfg, "recency", "recency", history_lines, candidates)


def run_vanilla_rag(client: LLMClient, cfg: Config, history_lines: list[str], candidates: list[dict]):
    return _run_ranking(client, cfg, "vanilla_rag", "vanilla_rag", history_lines, candidates)
