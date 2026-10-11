"""Demo recommender service: feed, search, feedback, profile, replay.

Wires together the LLM client, GRU scorer, FAISS index, ARAG agents, and SQLite.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pandas as pd

from arag import util
from arag.agents.arag import run_user_async
from arag.agents.context import build_context
from arag.agents.uua import run_uua
from arag.config import get_config
from arag.llm.client import LLMClient
from arag.memory import repo
from arag.models.score import GRUScorer
from arag.ranking.fusion import fuse


def evidence_snippet(text: str, limit: int = 180) -> str:
    """An exact substring of item text, selected in code (never generated)."""
    if not text:
        return ""
    first_sentence = text.split(". ")[0]
    if len(first_sentence) > limit:
        return text[:limit] + "..."
    return first_sentence


class RecommenderService:
    def __init__(self, cfg, db_path: str | Path):
        self.cfg = cfg
        self.db_path = str(db_path)
        self.client = LLMClient(cfg)
        self._gru = None
        self._embedder = None
        self._faiss_index = None
        self._emb_ids = None
        self._emb_matrix = None
        self._items_df = None
        self._items_index = None
        # demo GRU: the text-aware GRU when trained (app.gru_variant: text), else the ID-only GRU
        text_dir = self.cfg.path("gru_text") / "0"
        self.gru_variant = (
            "text" if self.cfg.get("app.gru_variant", "text") == "text" and (text_dir / "model.pt").exists() else "id"
        )
        self.gru_dir = text_dir if self.gru_variant == "text" else self.cfg.path("gru") / "0"
        self.alpha = self._load_alpha()
        self.theta = 6
        self._theta_and_alpha_from_results()

    # -- lazy resources ---------------------------------------------------
    def _load_alpha(self) -> float:
        p = self.cfg.path("results") / "metrics.json"
        if p.exists():
            try:
                return float(util.read_json(p).get(self._alpha_key(), 0.5))
            except Exception:
                return 0.5
        return 0.5

    def _theta_and_alpha_from_results(self) -> None:
        p = self.cfg.path("results") / "metrics.json"
        if p.exists():
            try:
                m = util.read_json(p)
                self.theta = int(m.get("best_theta", 6))
                self.alpha = float(m.get(self._alpha_key(), 0.5))
            except Exception:
                pass

    def _alpha_key(self) -> str:
        return "best_alpha_text" if self.gru_variant == "text" else "best_alpha"

    def _gru_model(self) -> GRUScorer:
        if self._gru is None:
            self._gru = GRUScorer(str(self.gru_dir))
        return self._gru

    def _embedder_model(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.cfg.get("embedding.model"))
        return self._embedder

    def _faiss(self):
        if self._faiss_index is None:
            import faiss
            faiss_dir = self.cfg.path("faiss")
            self._faiss_index = faiss.read_index(str(faiss_dir / "index.faiss"))
            self._emb_ids = np.load(faiss_dir / "item_ids.npy", allow_pickle=True)
            self._emb_matrix = np.load(faiss_dir / "item_embeddings.npy")
        return self._faiss_index

    def _items(self) -> pd.DataFrame:
        if self._items_df is None:
            self._items_df = pd.read_parquet(self.cfg.path("data_processed") / "items.parquet")
            self._items_index = self._items_df.set_index("item_id")
        return self._items_df

    def _item_row(self, item_id: str):
        self._items()
        if item_id in self._items_index.index:
            return self._items_index.loc[item_id]
        return None

    # -- helpers ----------------------------------------------------------
    def _history_df(self, conn, user_id: str) -> pd.DataFrame:
        rows = conn.execute(
            """SELECT i.user_id, i.item_id, i.ts AS timestamp, i.rating,
                      it.title, it.main_category
               FROM interactions i LEFT JOIN items it ON i.item_id = it.item_id
               WHERE i.user_id = ? ORDER BY i.ts, i.item_id""",
            (user_id,),
        ).fetchall()
        return pd.DataFrame(
            [dict(r) for r in rows],
            columns=["user_id", "item_id", "timestamp", "rating", "title", "main_category"],
        ).fillna({"title": "", "main_category": "", "rating": 0.0})

    def _candidate_items(self, item_ids: list[str]) -> list[dict]:
        out = []
        for iid in item_ids:
            row = self._item_row(iid)
            out.append(
                {
                    "item_id": iid,
                    "title": str(row["title"]) if row is not None else "",
                    "text": str(row["text"]) if row is not None else "",
                }
            )
        return out

    def _fused_scores(self, fused: list[str]) -> dict[str, float]:
        n = len(fused)
        if n <= 1:
            return {iid: 1.0 for iid in fused}
        return {iid: 1.0 - i / (n - 1) for i, iid in enumerate(fused)}

    def _trace_dict(self, trace, kind: str) -> dict:
        def title_of(iid):
            row = self._item_row(iid)
            return str(row["title"]) if row is not None else iid

        nli = []
        if trace.nli:
            for iid, d in sorted(trace.nli.items.items(), key=lambda kv: -kv[1]["score"]):
                nli.append(
                    {
                        "item_id": iid,
                        "title": title_of(iid),
                        "label": d["label"],
                        "score": d["score"],
                        "rationale": d.get("rationale", ""),
                        "accepted": d["score"] >= self.theta,
                    }
                )
        return {
            "kind": kind,
            "alpha": self.alpha,
            "theta": self.theta,
            "user_understanding": trace.s_user_text,
            "nli": nli,
            "nli_missing": trace.nli.nli_missing if trace.nli else 0,
            "nli_reasked": trace.nli.reasked if trace.nli else False,
            "context_summary": trace.s_ctx_text,
            "csa_empty_fallback": trace.csa_empty_fallback,
            "final_ranking": [{"item_id": iid, "title": title_of(iid)} for iid in trace.final_ranking],
            "reasons": trace.reasons,
            "fallbacks": trace.fallbacks,
            "repairs": {
                "unknown": trace.repairs.unknown,
                "duplicate": trace.repairs.duplicate,
                "missing": trace.repairs.missing,
            },
        }

    def _rank(self, user_id: str, history_df: pd.DataFrame, candidate_ids: list[str]):
        cand_items = self._candidate_items(candidate_ids)
        trace = asyncio.run(
            run_user_async(self.client, self.cfg, user_id, history_df, cand_items, self.theta)
        )
        hist_ids = history_df["item_id"].astype(str).tolist()
        gru_rank = self._gru_model().rank(hist_ids, candidate_ids)
        nli_scores = trace.nli.scores() if trace.nli else None
        fused = fuse(candidate_ids, trace.final_ranking, gru_rank, self.alpha, nli_scores)
        return fused, trace.reasons, trace

    def _render_items(self, item_ids: list[str], reasons: dict, scores: dict) -> list[dict]:
        out = []
        for iid in item_ids:
            row = self._item_row(iid)
            out.append(
                {
                    "item_id": iid,
                    "title": str(row["title"]) if row is not None else iid,
                    "category": str(row["main_category"]) if row is not None else "",
                    "price": float(row["price"]) if row is not None and pd.notna(row.get("price")) else None,
                    "reason": reasons.get(iid, ""),
                    "evidence": evidence_snippet(str(row["text"])) if row is not None else "",
                    "score": float(scores.get(iid, 0.0)),
                }
            )
        return out

    def _store_run(self, conn, user_id, session_id, kind, request_json, page_ids, reasons, scores, trace):
        run_id = repo.create_run(
            conn, user_id, session_id, kind, request_json,
            {"alpha": self.alpha, "theta": self.theta},
            trace=self._trace_dict(trace, kind) if trace is not None else None,
        )
        rec_items = []
        for i, iid in enumerate(page_ids):
            row = self._item_row(iid)
            text = str(row["text"]) if row is not None else ""
            rec_items.append(
                {
                    "item_id": iid,
                    "rank": i + 1,
                    "score": float(scores.get(iid, 0.0)),
                    "reason": {"reason": reasons.get(iid, ""), "evidence": evidence_snippet(text)},
                }
            )
        repo.add_rec_items(conn, run_id, rec_items)
        return run_id

    # -- public API -------------------------------------------------------
    def preload(self) -> None:
        """Eagerly load heavy resources in the main thread before serving requests."""
        self._items()
        for loader in (self._embedder_model, self._faiss, self._gru_model):
            try:
                loader()
            except Exception:
                pass

    def health(self) -> dict:
        gru_ok = (self.gru_dir / "model.pt").exists()
        faiss_ok = (self.cfg.path("faiss") / "index.faiss").exists()
        db_ok = Path(self.db_path).exists()
        return {
            "db": db_ok,
            "faiss": faiss_ok,
            "gru": gru_ok,
            "gru_variant": self.gru_variant,
            "alpha": self.alpha,
            "llm_key": self.cfg.llm_has_key(),
            "model": self.cfg.llm_model,
        }

    def list_users(self) -> list[dict]:
        conn = repo.connect(self.db_path)
        try:
            return repo.list_users(conn)
        finally:
            conn.close()

    def history(self, user_id: str, limit: int = 30) -> list[dict]:
        conn = repo.connect(self.db_path)
        try:
            return repo.get_history(conn, user_id, limit)
        finally:
            conn.close()

    def create_session(self, user_id: str, session_id: str | None = None) -> dict:
        conn = repo.connect(self.db_path)
        try:
            if not repo.user_exists(conn, user_id):
                raise KeyError("user not found")
            sid = repo.create_session(conn, user_id, session_id)
            return {"session_id": sid, "user_id": user_id}
        finally:
            conn.close()

    def _check_session(self, conn, session_id: str, user_id: str) -> None:
        sess = repo.get_session(conn, session_id)
        if sess is None:
            raise KeyError("session not found")
        if sess["user_id"] != user_id:
            raise PermissionError("session does not belong to user")

    def feed(self, user_id: str, session_id: str, page: int, page_size: int) -> dict:
        conn = repo.connect(self.db_path)
        try:
            self._check_session(conn, session_id, user_id)
            history = self._history_df(conn, user_id)
            candidate_ids = self._retrieve_feed(conn, user_id, history)
            if not candidate_ids:
                return {"run_id": "", "user_id": user_id, "page": page, "page_size": page_size, "items": []}
            fused, reasons, trace = self._rank(user_id, history, candidate_ids)
            scores = self._fused_scores(fused)
            start = page * page_size
            page_ids = fused[start : start + page_size]
            items = self._render_items(page_ids, reasons, scores)
            run_id = self._store_run(conn, user_id, session_id, "feed",
                                     {"page": page, "page_size": page_size},
                                     page_ids, reasons, scores, trace)
            return {"run_id": run_id, "user_id": user_id, "page": page, "page_size": page_size, "items": items}
        finally:
            conn.close()

    def _filter_candidates(self, conn, user_id: str, ids: list[str], cap: int) -> list[str]:
        interacted = repo.interacted_items(conn, user_id) | repo.dismissed_items(conn, user_id)
        seen = set()
        out = []
        for iid in ids:
            if iid in seen or iid in interacted:
                continue
            seen.add(iid)
            out.append(iid)
            if len(out) >= cap:
                break
        return out

    def _retrieve_feed(self, conn, user_id: str, history: pd.DataFrame, k: int = 50) -> list[str]:
        ids = []
        titles = " ".join(history["title"].astype(str).tolist()[-20:])
        if titles.strip():
            emb = self._embedder_model().encode([titles], normalize_embeddings=True).astype(np.float32)
            idx = self._faiss()
            _, I = idx.search(emb, k)
            ids.extend(self._emb_ids[I[0]].tolist())
        gru = self._gru_model()
        hist_ids = history["item_id"].astype(str).tolist()
        if hist_ids:
            h_u = gru.user_vector(hist_ids)
            import torch
            with torch.no_grad():
                logits = h_u @ gru.model.all_item_emb().T
                topk = torch.topk(logits, k).indices.tolist()
            idx2item = {v: kk for kk, v in gru.item2idx.items()}
            for t in topk:
                if t in idx2item:
                    ids.append(idx2item[t])
        return self._filter_candidates(conn, user_id, ids, 20)

    def search(self, user_id: str, session_id: str, query: str, filters: dict | None, limit: int) -> dict:
        conn = repo.connect(self.db_path)
        try:
            self._check_session(conn, session_id, user_id)
            history = self._history_df(conn, user_id)
            q = self._parse_query(query)
            candidate_ids = self._retrieve_search(conn, user_id, q, filters)
            if not candidate_ids:
                return {"run_id": "", "items": []}
            fused, reasons, trace = self._rank(user_id, history, candidate_ids[:20])
            scores = self._fused_scores(fused)
            top = fused[:limit]
            items = self._render_items(top, reasons, scores)
            run_id = self._store_run(conn, user_id, session_id, "search",
                                     {"query": query, "filters": filters, "limit": limit},
                                     top, reasons, scores, trace)
            return {"run_id": run_id, "items": items}
        finally:
            conn.close()

    def _parse_query(self, query: str) -> dict:
        from arag.agents.prompts import load_prompt, render
        from arag.llm.schemas import SearchQuery, to_strict_schema
        prompt_path = self.cfg.path("prompts") / "search.txt"
        if not prompt_path.exists():
            return {"query": query, "keywords": query.split(), "max_price": None, "exclude_terms": []}
        template = load_prompt(self.cfg, "search")
        prompt = render(template, QUERY=query)
        schema = to_strict_schema(SearchQuery, "search_query")
        res = self.client.complete_validated(
            "search", prompt, schema, SearchQuery,
            fallback=SearchQuery(keywords=query.split(), max_price=None, exclude_terms=[]),
        )
        q = res.obj
        return {"query": query, "keywords": q.keywords, "max_price": q.max_price, "exclude_terms": q.exclude_terms}

    def _retrieve_search(self, conn, user_id: str, q: dict, filters: dict | None, k: int = 50) -> list[str]:
        items = self._items()
        mask = items["has_text"].astype(bool)
        mx = q.get("max_price") or (filters or {}).get("max_price")
        if mx is not None:
            mask &= items["price"].fillna(float("inf")) <= mx
        for term in q.get("exclude_terms", []):
            mask &= ~items["text"].str.contains(term, case=False, na=False)
        eligible = set(items.loc[mask, "item_id"].astype(str))
        emb = self._embedder_model().encode(
            [q.get("query", " ".join(q["keywords"]))], normalize_embeddings=True
        ).astype(np.float32)
        idx = self._faiss()
        _, I = idx.search(emb, k * 5)
        candidate_ids = [iid for iid in self._emb_ids[I[0]].tolist() if iid in eligible]
        return self._filter_candidates(conn, user_id, candidate_ids, k)

    def feedback(self, user_id: str, session_id: str, item_id: str, event_type: str) -> None:
        conn = repo.connect(self.db_path)
        try:
            self._check_session(conn, session_id, user_id)
            repo.add_interaction(conn, user_id, item_id, event_type, source="app")
            repo.add_session_event(conn, session_id, user_id, item_id, event_type)
        finally:
            conn.close()

    def get_profile(self, user_id: str) -> dict | None:
        conn = repo.connect(self.db_path)
        try:
            return repo.get_profile(conn, user_id)
        finally:
            conn.close()

    def delete_profile(self, user_id: str) -> None:
        conn = repo.connect(self.db_path)
        try:
            repo.delete_profile(conn, user_id)
        finally:
            conn.close()

    def generate_profile(self, user_id: str) -> dict:
        conn = repo.connect(self.db_path)
        try:
            history = self._history_df(conn, user_id)
            c_st, c_lt = build_context(
                history,
                self.cfg.get("splits.session_context", 5),
                self.cfg.get("splits.long_term_context", 50),
            )
            uua, _ = run_uua(self.client, self.cfg, c_lt, c_st)
            event_ids = [
                int(e["event_id"]) for e in conn.execute(
                    "SELECT event_id FROM interactions WHERE user_id=?", (user_id,)
                ).fetchall()
            ]
            prev = repo.get_profile(conn, user_id)
            version = (prev["version"] + 1) if prev else 1
            provenance = {
                "model": self.cfg.llm_model,
                "events_used": len(event_ids),
                "event_ids": event_ids[-50:],
            }
            repo.save_profile(conn, user_id, uua.model_dump(), provenance, version)
            return repo.get_profile(conn, user_id)
        finally:
            conn.close()

    def get_run(self, run_id: str) -> dict | None:
        conn = repo.connect(self.db_path)
        try:
            return repo.get_run(conn, run_id)
        finally:
            conn.close()

    def replay(self, run_id: str) -> dict:
        run = self.get_run(run_id)
        if run is None:
            raise KeyError("run not found")
        req = run["request_json"]
        if run["kind"] == "feed":
            return self.feed(run["user_id"], run["session_id"], req.get("page", 0), req.get("page_size", 10))
        if run["kind"] == "search":
            return self.search(run["user_id"], run["session_id"], req.get("query", ""), req.get("filters"), req.get("limit", 5))
        raise KeyError("unknown run kind")
