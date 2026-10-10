"""Orchestrated ARAG pipeline: UUA || NLI -> CSA -> IRA."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pandas as pd

from arag.agents.context import build_context
from arag.agents.csa import run_csa
from arag.agents.ira import run_ira
from arag.agents.nli import NLIResult, run_nli, select_accepted
from arag.agents.uua import run_uua, uua_to_text
from arag.config import Config
from arag.llm.client import LLMClient
from arag.ranking.parser import RepairStats, parse_ranking


@dataclass
class ARAGTrace:
    user_id: str
    s_user_text: str = ""
    nli: NLIResult | None = None
    s_ctx_text: str = ""
    csa_empty_fallback: bool = False
    final_ranking: list[str] = field(default_factory=list)
    reasons: dict = field(default_factory=dict)
    repairs: RepairStats = field(default_factory=RepairStats)
    fallbacks: dict = field(default_factory=dict)


async def run_user_async(
    client: LLMClient,
    cfg: Config,
    user_id: str,
    history: pd.DataFrame,
    candidate_items: list[dict],
    theta: int,
    semaphore: asyncio.Semaphore | None = None,
) -> ARAGTrace:
    async def _run():
        c_st, c_lt = build_context(
            history,
            n_session=cfg.get("splits.session_context", 5),
            n_long=cfg.get("splits.long_term_context", 50),
        )
        raw_context = "\n".join(c_lt + c_st)

        # UUA || NLI (parallel)
        uua_task = asyncio.to_thread(run_uua, client, cfg, c_lt, c_st)
        nli_task = asyncio.to_thread(run_nli, client, cfg, raw_context, candidate_items)
        (uua, uua_res), nli = await asyncio.gather(uua_task, nli_task)

        s_user = uua_to_text(uua)

        # CSA: accepted = score >= theta; empty fallback -> top 3 by score
        accepted, csa_empty_fallback = select_accepted(nli, theta)

        item_by_id = {c["item_id"]: c for c in candidate_items}
        accepted_items = [
            {"item_id": iid, "score": d["score"], "text": item_by_id.get(iid, {}).get("text", "")}
            for iid, d in accepted
        ]

        csa, csa_res, _ = await asyncio.to_thread(
            run_csa, client, cfg, s_user, accepted_items, csa_empty_fallback
        )

        ira, ira_res = await asyncio.to_thread(
            run_ira, client, cfg, s_user, csa.summary, candidate_items
        )

        candidate_ids = [c["item_id"] for c in candidate_items]
        final_ranking, repairs = parse_ranking(
            list(ira.ranking), candidate_ids, nli.scores()
        )

        return ARAGTrace(
            user_id=user_id,
            s_user_text=s_user,
            nli=nli,
            s_ctx_text=csa.summary,
            csa_empty_fallback=csa_empty_fallback,
            final_ranking=final_ranking,
            reasons={r.item_id: r.reason for r in ira.reasons},
            repairs=repairs,
            fallbacks={
                "uua": uua_res.fallback,
                "nli": nli.fallback_calls,
                "csa": csa_res.fallback,
                "ira": ira_res.fallback,
            },
        )

    if semaphore is not None:
        async with semaphore:
            return await _run()
    return await _run()


async def run_users_async(
    client: LLMClient,
    cfg: Config,
    user_histories: dict[str, pd.DataFrame],
    user_candidates: dict[str, list[dict]],
    theta: int,
    concurrency: int = 8,
) -> list[ARAGTrace]:
    semaphore = asyncio.Semaphore(concurrency)
    tasks = []
    for user_id in user_histories:
        tasks.append(
            run_user_async(
                client,
                cfg,
                user_id,
                user_histories[user_id],
                user_candidates[user_id],
                theta,
                semaphore,
            )
        )
    return await asyncio.gather(*tasks)
