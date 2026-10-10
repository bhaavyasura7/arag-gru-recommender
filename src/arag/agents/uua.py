"""User Understanding Agent (UUA)."""
from __future__ import annotations

from arag.agents.prompts import load_prompt, render
from arag.config import Config
from arag.llm.client import LLMClient
from arag.llm.schemas import UUAOutput, to_strict_schema


def fallback_uua() -> UUAOutput:
    return UUAOutput(long_term_interests=[], current_intent="", avoid=[], summary="")


def run_uua(client: LLMClient, cfg: Config, c_lt: list[str], c_st: list[str]):
    template = load_prompt(cfg, "uua")
    prompt = render(template, C_LT="\n".join(c_lt), C_ST="\n".join(c_st))
    schema = to_strict_schema(UUAOutput, "uua")
    res = client.complete_validated("uua", prompt, schema, UUAOutput, fallback=fallback_uua())
    return res.obj, res


def uua_to_text(uua: UUAOutput) -> str:
    parts = [
        f"long_term_interests: {uua.long_term_interests}",
        f"current_intent: {uua.current_intent}",
        f"avoid: {uua.avoid}",
        f"summary: {uua.summary}",
    ]
    return "\n".join(parts)
