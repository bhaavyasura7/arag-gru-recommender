"""Evaluation driver: validation (theta/alpha tuning) then test, all systems.

Resumable via the LLM disk cache. Saves per-user predictions and a run record.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from arag import util
from arag.agents.arag import run_users_async
from arag.agents.baselines import run_recency, run_vanilla_rag
from arag.agents.context import build_context, history_before, render_lines
from arag.agents.csa import run_csa
from arag.agents.ira import run_ira
from arag.agents.nli import run_nli, select_accepted
from arag.agents.uua import run_uua, uua_to_text
from arag.config import get_config
from arag.eval.bootstrap import paired_bootstrap
from arag.eval.metrics import hit_at_k, mrr, ndcg_at_k, rank_of
from arag.llm.client import LLMClient
from arag.models.score import GRUScorer
from arag.ranking.content import ItemKNN, content_ranking
from arag.ranking.fusion import fuse
from arag.ranking.parser import parse_ranking
from arag.ranking.simple import popularity_ranking, random_ranking, seed_for

METRICS = {
    "hit@5": lambda r: hit_at_k(r, 5),
    "ndcg@5": lambda r: ndcg_at_k(r, 5),
    "hit@10": lambda r: hit_at_k(r, 10),
    "ndcg@10": lambda r: ndcg_at_k(r, 10),
    "mrr": mrr,
}


def per_user_metrics(ranked_ids: list[str], target: str) -> dict[str, float]:
    rank = rank_of(target, ranked_ids)
    return {name: fn(rank) for name, fn in METRICS.items()}


class EvalData:
    def __init__(self, cfg):
        self.cfg = cfg
        proc = cfg.path("data_processed")
        lists = cfg.path("lists")
        self.items = pd.read_parquet(proc / "items.parquet")
        self.interactions = pd.read_parquet(proc / "interactions.parquet")
        self.sequences = pd.read_parquet(lists / "sequences.parquet")
        self.valid_lists = pd.read_parquet(lists / "valid_lists.parquet")
        self.test_lists = pd.read_parquet(lists / "test_lists.parquet")
        self.users = util.read_json(lists / "users.json")

        self.title = dict(zip(self.items["item_id"].astype(str), self.items["title"].astype(str)))
        self.text = dict(zip(self.items["item_id"].astype(str), self.items["text"].astype(str)))
        train = self.interactions[self.interactions["split"] == "train"]
        self.popularity = train["item_id"].astype(str).value_counts().to_dict()

        self.seq_by_user = {
            row["user_id"]: row for _, row in self.sequences.iterrows()
        }

        hist = self.interactions.merge(
            self.items[["item_id", "title", "main_category"]], on="item_id", how="left"
        )
        hist["title"] = hist["title"].fillna("")
        hist["main_category"] = hist["main_category"].fillna("")
        hist = hist.sort_values(["user_id", "timestamp", "item_id"])
        self.history = {
            u: g.reset_index(drop=True) for u, g in hist.groupby("user_id")
        }

        self.emb_ids = None
        self.emb = None

    def history_for(self, user_id: str, split: str) -> pd.DataFrame:
        h = self.history.get(user_id)
        if h is None:
            return pd.DataFrame(columns=["item_id", "title", "main_category", "rating", "timestamp", "split"])
        row = self.seq_by_user.get(user_id)
        if row is None:
            return h[h["split"] == "train"]
        target_ts = int(row["valid_ts"] if split == "valid" else row["test_ts"])
        target_id = str(row["valid_item"] if split == "valid" else row["test_item"])
        return history_before(h, target_ts, target_id)

    def candidate_items(self, split: str, user_id: str) -> list[dict]:
        lists = self.valid_lists if split == "valid" else self.test_lists
        row = lists[lists["user_id"] == user_id].iloc[0]
        return [
            {"item_id": iid, "title": self.title.get(iid, ""), "text": self.text.get(iid, "")}
            for iid in row["candidates"]
        ]

    def load_embeddings(self):
        if self.emb is not None:
            return
        faiss_dir = self.cfg.path("faiss")
        self.emb_ids = np.load(faiss_dir / "item_ids.npy", allow_pickle=True)
        self.emb = np.load(faiss_dir / "item_embeddings.npy")
        self.emb_idx = {iid: i for i, iid in enumerate(self.emb_ids)}


async def run_llm_systems_user(client, cfg, data: EvalData, user_id, split, theta, semaphore) -> dict:
    async with semaphore:
        history = data.history_for(user_id, split)
        cand_items = data.candidate_items(split, user_id)
        cand_ids = [c["item_id"] for c in cand_items]
        target = str(data.seq_by_user[user_id][f"{split}_item"])

        c_st, c_lt = build_context(
            history, n_session=cfg.get("splits.session_context", 5), n_long=cfg.get("splits.long_term_context", 50)
        )
        raw_ctx = "\n".join(c_lt + c_st)

        uua, _ = await asyncio.to_thread(run_uua, client, cfg, c_lt, c_st)
        nli = await asyncio.to_thread(run_nli, client, cfg, raw_ctx, cand_items)
        s_user = uua_to_text(uua)

        # ARAG
        accepted, empty_fb = select_accepted(nli, theta)
        acc_items = [
            {"item_id": iid, "score": d["score"], "text": data.text.get(iid, "")}
            for iid, d in accepted
        ]
        csa, _, _ = await asyncio.to_thread(run_csa, client, cfg, s_user, acc_items, empty_fb)
        ira, _ = await asyncio.to_thread(run_ira, client, cfg, s_user, csa.summary, cand_items)
        arag_ranking, _ = parse_ranking(list(ira.ranking), cand_ids, nli.scores())

        # ablation: w/o NLI (CSA over all 20 candidates)
        all_items = [{"item_id": c["item_id"], "score": 0, "text": c["text"]} for c in cand_items]
        csa2, _, _ = await asyncio.to_thread(run_csa, client, cfg, s_user, all_items, False)
        ira2, _ = await asyncio.to_thread(run_ira, client, cfg, s_user, csa2.summary, cand_items)
        abl_nli, _ = parse_ranking(list(ira2.ranking), cand_ids, nli.scores())

        # ablation: w/o NLI & CSA (IRA with raw candidates)
        ira3, _ = await asyncio.to_thread(run_ira, client, cfg, s_user, "", cand_items)
        abl_both, _ = parse_ranking(list(ira3.ranking), cand_ids, nli.scores())

        # LLM-Recency: last 10 history lines
        hist_lines = render_lines(history)[-10:]
        rec, _ = await asyncio.to_thread(run_recency, client, cfg, hist_lines, cand_items)
        recency, _ = parse_ranking(list(rec.ranking), cand_ids)

        # LLM-Vanilla-RAG: 10 history items most similar to mean candidate embedding
        similar = select_similar_history(data, history, cand_items, k=10)
        rag, _ = await asyncio.to_thread(run_vanilla_rag, client, cfg, similar, cand_items)
        vanilla_rag, _ = parse_ranking(list(rag.ranking), cand_ids)

        return {
            "user_id": user_id,
            "target": target,
            "candidates": cand_ids,
            "nli_scores": nli.scores(),
            "arag": arag_ranking,
            "arag_wo_nli": abl_nli,
            "arag_wo_nli_csa": abl_both,
            "recency": recency,
            "vanilla_rag": vanilla_rag,
        }


def select_similar_history(data: EvalData, history: pd.DataFrame, cand_items: list[dict], k: int = 10) -> list[str]:
    data.load_embeddings()
    if data.emb is None:
        return render_lines(history)[-k:]
    cand_emb = np.stack([data.emb[data.emb_idx[i]] for i in [c["item_id"] for c in cand_items] if i in data.emb_idx])
    if cand_emb.size == 0:
        return render_lines(history)[-k:]
    mean_cand = cand_emb.mean(axis=0)
    hist_ids = history["item_id"].astype(str).tolist()
    rows = []
    for iid in hist_ids:
        if iid in data.emb_idx:
            sim = float(data.emb[data.emb_idx[iid]] @ mean_cand)
            rows.append((sim, iid))
    rows.sort(key=lambda x: -x[0])
    top_ids = [iid for _, iid in rows[:k]]
    top_set = set(top_ids)
    selected = history[history["item_id"].astype(str).isin(top_set)]
    lines = render_lines(selected)
    return lines[:k]


def compute_system_predictions(
    rankings: dict[str, dict[str, list[str]]],
    targets: dict[str, str],
) -> pd.DataFrame:
    rows = []
    for user_id in rankings:
        for system in rankings[user_id]:
            ranked = rankings[user_id][system]
            target = targets[user_id]
            rows.append(
                {
                    "user_id": user_id,
                    "system": system,
                    "ranked_ids": ranked,
                    "target": target,
                    "rank": rank_of(target, ranked),
                }
            )
    return pd.DataFrame(rows)


def save_predictions(
    cfg,
    system: str,
    split: str,
    rankings: dict[str, list[str]],
    targets: dict[str, str],
    csv_users: set[str] | None = None,
) -> None:
    pred_dir = cfg.path("predictions")
    rows = [
        {"user_id": u, "ranked_ids": rankings[u], "target": targets[u], "rank": rank_of(targets[u], rankings[u])}
        for u in rankings
    ]
    df = pd.DataFrame(rows)
    df.to_parquet(pred_dir / f"{system}_{split}.parquet", index=False)
    csv_dir = cfg.path("predictions_csv")
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_df = df[df["user_id"].isin(csv_users)] if csv_users is not None else df
    csv_df.to_csv(csv_dir / f"{system}_{split}.csv", index=False)


def main() -> None:
    logger = util.setup_logging("arag.eval.run")
    cfg = get_config()
    data = EvalData(cfg)
    client = LLMClient(cfg)
    pred_dir: Path = cfg.path("predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    # ---- Phase 1: validation (theta + alpha tuning) ----
    val_users = data.users["validation"]
    val_rankings: dict[str, dict[str, list[str]]] = {}

    # GRU ranking for validation (seed 0)
    gru = GRUScorer(str(cfg.path("gru") / "0"))
    val_gru: dict[str, list[str]] = {}
    for u in val_users:
        hist = data.history_for(u, "valid")["item_id"].astype(str).tolist()
        cand_ids = [c["item_id"] for c in data.candidate_items("valid", u)]
        val_gru[u] = gru.rank(hist, cand_ids)

    # ARAG UUA + NLI for validation (theta-independent)
    val_candidates = {u: data.candidate_items("valid", u) for u in val_users}
    val_histories = {u: data.history_for(u, "valid") for u in val_users}
    sem = asyncio.Semaphore(cfg.get("llm.concurrency", 8))

    async def run_val_uua_nli():
        results = {}
        async def one(u):
            async with sem:
                h = val_histories[u]
                c_st, c_lt = build_context(h, cfg.get("splits.session_context", 5), cfg.get("splits.long_term_context", 50))
                raw = "\n".join(c_lt + c_st)
                uua, _ = await asyncio.to_thread(run_uua, client, cfg, c_lt, c_st)
                nli = await asyncio.to_thread(run_nli, client, cfg, raw, val_candidates[u])
                return u, uua_to_text(uua), nli
        for u, s_user, nli in await asyncio.gather(*[one(u) for u in val_users]):
            results[u] = (s_user, nli)
        return results

    val_uua_nli = asyncio.run(run_val_uua_nli())
    logger.info("validation UUA+NLI done (%d users)", len(val_uua_nli))

    # theta tuning
    theta_scores = {}
    theta_rankings = {}
    for theta in cfg.get("agents.theta_grid", [4, 5, 6, 7]):
        rankings = {}
        async def run_theta():
            sem2 = asyncio.Semaphore(cfg.get("llm.concurrency", 8))
            async def one(u):
                async with sem2:
                    s_user, nli = val_uua_nli[u]
                    accepted, empty_fb = select_accepted(nli, theta)
                    acc_items = [{"item_id": iid, "score": d["score"], "text": data.text.get(iid, "")} for iid, d in accepted]
                    csa, _, _ = await asyncio.to_thread(run_csa, client, cfg, s_user, acc_items, empty_fb)
                    ira, _ = await asyncio.to_thread(run_ira, client, cfg, s_user, csa.summary, val_candidates[u])
                    ranking, _ = parse_ranking(list(ira.ranking), [c["item_id"] for c in val_candidates[u]], nli.scores())
                    return u, ranking
            for u, r in await asyncio.gather(*[one(u) for u in val_users]):
                rankings[u] = r
            return rankings
        rankings = asyncio.run(run_theta())
        theta_rankings[theta] = rankings
        ndcgs = [ndcg_at_k(rank_of(str(data.seq_by_user[u]["valid_item"]), rankings[u]), 5) for u in val_users]
        theta_scores[theta] = float(np.mean(ndcgs))
        logger.info("theta=%d validation NDCG@5=%.4f", theta, theta_scores[theta])

    best_theta = max(theta_scores, key=theta_scores.get)
    logger.info("best theta=%d", best_theta)
    val_arag = theta_rankings[best_theta]

    # alpha tuning (validation, seed-0 GRU)
    val_targets = {u: str(data.seq_by_user[u]["valid_item"]) for u in val_users}
    alpha_curve_val = {}
    for alpha in cfg.get("fusion.alpha_grid"):
        fused = {}
        for u in val_users:
            cand_ids = [c["item_id"] for c in val_candidates[u]]
            fused[u] = fuse(cand_ids, val_arag[u], val_gru[u], alpha, val_uua_nli[u][1].scores())
        ndcgs = [ndcg_at_k(rank_of(val_targets[u], fused[u]), 5) for u in val_users]
        alpha_curve_val[alpha] = float(np.mean(ndcgs))
    best_alpha = max(alpha_curve_val, key=alpha_curve_val.get)
    logger.info("best alpha=%.1f", best_alpha)

    save_predictions(cfg, "arag", "valid", val_arag, val_targets)
    save_predictions(cfg, "gru", "valid", val_gru, val_targets)

    # content / ItemKNN baselines: recency decay tuned on ALL validation users (no LLM)
    data.load_embeddings()
    knn = ItemKNN([list(r["train_items"]) for r in data.seq_by_user.values()])
    val_all = data.valid_lists
    val_all_hist = {u: data.history_for(u, "valid")["item_id"].astype(str).tolist() for u in val_all["user_id"]}
    val_all_cands = dict(zip(val_all["user_id"], (list(c) for c in val_all["candidates"])))
    val_all_targets = {u: str(data.seq_by_user[u]["valid_item"]) for u in val_all["user_id"]}

    def tune_decay(rank_fn) -> tuple[float, dict[str, float]]:
        curve = {}
        for d in cfg.get("content.decay_grid", [0.0, 0.3, 0.6, 1.0]):
            ndcgs = [ndcg_at_k(rank_of(val_all_targets[u], rank_fn(val_all_hist[u], c, d)), 5) for u, c in val_all_cands.items()]
            curve[d] = float(np.mean(ndcgs))
        return max(curve, key=curve.get), curve

    def content_fn(h, c, d):
        return content_ranking(h, c, data.emb, data.emb_idx, d)

    best_decay_content, decay_curve_content = tune_decay(content_fn)
    best_decay_knn, decay_curve_knn = tune_decay(knn.rank)
    logger.info("content decay=%.1f %s | itemknn decay=%.1f %s", best_decay_content, decay_curve_content, best_decay_knn, decay_curve_knn)

    # alpha for ARAG + content fusion (100 validation users with cached ARAG output)
    val_content = {u: content_fn(val_all_hist[u], val_all_cands[u], best_decay_content) for u in val_users}
    alpha_curve_val_content = {}
    for alpha in cfg.get("fusion.alpha_grid"):
        ndcgs = [
            ndcg_at_k(rank_of(val_targets[u], fuse(val_all_cands[u], val_arag[u], val_content[u], alpha, val_uua_nli[u][1].scores())), 5)
            for u in val_users
        ]
        alpha_curve_val_content[alpha] = float(np.mean(ndcgs))
    best_alpha_content = max(alpha_curve_val_content, key=alpha_curve_val_content.get)
    logger.info("best alpha (ARAG+content)=%.1f", best_alpha_content)

    # ---- Phase 2: test ----
    test_users = data.users["test"]
    test_lists_all = data.test_lists
    all_test_users = test_lists_all["user_id"].tolist()

    # non-LLM systems on ALL test users
    gru_scorers = {s: GRUScorer(str(cfg.path("gru") / str(s))) for s in cfg.get("gru.seeds", [0, 1, 2])}
    rand_all, pop_all, gru_all_seeds = {}, {}, {s: {} for s in cfg.get("gru.seeds", [0, 1, 2])}
    content_all, knn_all = {}, {}
    for u in all_test_users:
        cand_ids = list(test_lists_all[test_lists_all["user_id"] == u].iloc[0]["candidates"])
        rand_all[u] = random_ranking(cand_ids, seed_for(cfg.seed, "random", u))
        pop_all[u] = popularity_ranking(cand_ids, data.popularity)
        hist = data.history_for(u, "test")["item_id"].astype(str).tolist()
        content_all[u] = content_fn(hist, cand_ids, best_decay_content)
        knn_all[u] = knn.rank(hist, cand_ids, best_decay_knn)
        for s in cfg.get("gru.seeds", [0, 1, 2]):
            gru_all_seeds[s][u] = gru_scorers[s].rank(hist, cand_ids)

    # LLM systems on 100 test users
    sem_test = asyncio.Semaphore(cfg.get("llm.concurrency", 8))
    async def run_test():
        tasks = [
            run_llm_systems_user(client, cfg, data, u, "test", best_theta, sem_test)
            for u in test_users
        ]
        return await asyncio.gather(*tasks)

    llm_results = asyncio.run(run_test())
    logger.info("test LLM systems done (%d users)", len(llm_results))

    test_targets = {u: str(data.seq_by_user[u]["test_item"]) for u in all_test_users}
    test_cand_ids = {u: list(test_lists_all[test_lists_all["user_id"] == u].iloc[0]["candidates"]) for u in all_test_users}

    # build per-system rankings dicts for test (100 users for LLM, all for non-LLM)
    systems: dict[str, dict[str, list[str]]] = {
        "random": rand_all,
        "popularity": pop_all,
        "content": content_all,
        "itemknn": knn_all,
    }
    for s in cfg.get("gru.seeds", [0, 1, 2]):
        systems[f"gru_seed{s}"] = gru_all_seeds[s]

    arag_test, abl_nli_test, abl_both_test, recency_test, vrag_test = {}, {}, {}, {}, {}
    gru_seed0 = systems["gru_seed0"]
    for r in llm_results:
        u = r["user_id"]
        arag_test[u] = r["arag"]
        abl_nli_test[u] = r["arag_wo_nli"]
        abl_both_test[u] = r["arag_wo_nli_csa"]
        recency_test[u] = r["recency"]
        vrag_test[u] = r["vanilla_rag"]

    systems["recency"] = recency_test
    systems["vanilla_rag"] = vrag_test
    systems["arag"] = arag_test
    systems["arag_wo_nli"] = abl_nli_test
    systems["arag_wo_nli_csa"] = abl_both_test

    # fused (ARAG + GRU, seed 0) at best alpha
    fused_test = {}
    test_nli = {r["user_id"]: r["nli_scores"] for r in llm_results}
    for u in test_users:
        cand_ids = test_cand_ids[u]
        fused_test[u] = fuse(cand_ids, arag_test[u], gru_seed0[u], best_alpha, test_nli[u])
    systems["arag_gru"] = fused_test
    systems["arag_content"] = {
        u: fuse(test_cand_ids[u], arag_test[u], content_all[u], best_alpha_content, test_nli[u]) for u in test_users
    }

    # save predictions
    # save predictions (tracked CSVs capped to the 100 test users to keep results/ small)
    csv_users = set(test_users)
    for sys in ["recency", "vanilla_rag", "arag", "arag_wo_nli", "arag_wo_nli_csa", "arag_gru", "arag_content"]:
        save_predictions(cfg, sys, "test", systems[sys], test_targets, csv_users)
    for sys in ["random", "popularity", "content", "itemknn", "gru_seed0", "gru_seed1", "gru_seed2"]:
        save_predictions(cfg, sys, "test", systems[sys], test_targets, csv_users)

    # ---- metrics + bootstrap ----
    # Main table: every system on the same 100 sampled test users.
    sampled = set(test_users)
    systems_sampled = {sys: {u: r for u, r in rk.items() if u in sampled} for sys, rk in systems.items()}
    metrics = compute_all_metrics(systems_sampled, test_targets, cfg)
    # Secondary table: non-LLM systems on all test users.
    non_llm = ["random", "popularity", "content", "itemknn"] + [f"gru_seed{s}" for s in cfg.get("gru.seeds", [0, 1, 2])]
    metrics_all_users = {
        sys: v for sys, v in compute_all_metrics(
            {sys: systems[sys] for sys in non_llm}, test_targets, cfg
        ).items() if "_vs_" not in sys
    }

    # alpha curve on test
    alpha_curve_test = {}
    for alpha in cfg.get("fusion.alpha_grid"):
        fused = {u: fuse(test_cand_ids[u], arag_test[u], gru_seed0[u], alpha, test_nli[u]) for u in test_users}
        ndcgs = [ndcg_at_k(rank_of(test_targets[u], fused[u]), 5) for u in test_users]
        alpha_curve_test[alpha] = float(np.mean(ndcgs))

    elapsed = time.time() - t0
    result = {
        "theta_grid": theta_scores,
        "best_theta": best_theta,
        "best_alpha": best_alpha,
        "best_alpha_content": best_alpha_content,
        "alpha_curve_validation_content": {str(k): v for k, v in alpha_curve_val_content.items()},
        "content_decay": {"best": best_decay_content, "curve": {str(k): v for k, v in decay_curve_content.items()}},
        "itemknn_decay": {"best": best_decay_knn, "curve": {str(k): v for k, v in decay_curve_knn.items()}},
        "alpha_curve_validation": {str(k): v for k, v in alpha_curve_val.items()},
        "alpha_curve_test": {str(k): v for k, v in alpha_curve_test.items()},
        "metrics": metrics,
        "metrics_all_users": metrics_all_users,
        "n_test_users_sampled": len(test_users),
        "n_test_users_all": len(all_test_users),
        "cost": cost_summary(client),
        "elapsed_s": elapsed,
    }
    util.write_json(cfg.path("results") / "metrics.json", result)
    util.write_json(cfg.path("results") / "run_record.json", build_run_record(cfg, client, elapsed))
    logger.info("evaluation done in %.0fs; cost %.4f USD", elapsed, client.cost_usd())


def compute_all_metrics(systems: dict[str, dict[str, list[str]]], targets: dict[str, str], cfg) -> dict:
    ref_systems = ["arag", "gru_seed0"]

    # precompute per-user metric values per system (once)
    per_user: dict[str, dict[str, dict[str, float]]] = {}
    for sys, rankings in systems.items():
        vals: dict[str, dict[str, float]] = {m: {} for m in METRICS}
        for u, ranked in rankings.items():
            pm = per_user_metrics(ranked, targets[u])
            for m in METRICS:
                vals[m][u] = pm[m]
        per_user[sys] = vals

    out: dict = {}
    for sys in systems:
        out[sys] = {
            m: {
                "mean": float(np.mean(list(per_user[sys][m].values()))),
                "std": float(np.std(list(per_user[sys][m].values()))),
            }
            for m in METRICS
        }

    for sys in systems:
        if sys in ref_systems:
            continue
        for ref in ref_systems:
            if ref not in systems:
                continue  # subset call (e.g. non-LLM-only block) without this reference
            users = sorted(set(systems[sys]) & set(systems[ref]))
            if not users:
                continue
            for m in METRICS:
                a = [per_user[sys][m][u] for u in users]
                b = [per_user[ref][m][u] for u in users]
                md, lo, hi = paired_bootstrap(a, b, cfg.get("eval.bootstrap_resamples", 10000))
                out.setdefault(f"{sys}_vs_{ref}", {})[m] = {"mean_diff": md, "ci": [lo, hi]}
    return out


def cost_summary(client: LLMClient) -> dict:
    import collections
    per_agent = collections.defaultdict(lambda: {"calls": 0, "cache_hits": 0, "fallbacks": 0, "in": 0, "out": 0, "s": 0.0})
    for r in client.records:
        a = per_agent[r.agent]
        a["calls"] += 1
        a["cache_hits"] += int(r.cache_hit)
        a["fallbacks"] += int(r.fallback)
        a["in"] += r.prompt_tokens
        a["out"] += r.completion_tokens
        a["s"] += r.latency_s
    return {
        "cost_usd": client.cost_usd(),
        "tokens": client.total_tokens(),
        "per_agent": {k: dict(v) for k, v in per_agent.items()},
    }


def build_run_record(cfg, client, elapsed_s) -> dict:
    import subprocess
    commit = ""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    manifest = util.read_json(cfg.path("data_processed").parent / "manifest.json")
    return {
        "git_commit": commit,
        "model": cfg.llm_model,
        "provider": cfg.llm_provider,
        "category": cfg.category,
        "seed": cfg.seed,
        "elapsed_s": elapsed_s,
        "manifest": manifest,
        "config": cfg.raw,
    }


if __name__ == "__main__":
    main()
