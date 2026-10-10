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
        self._item_text = None
        self.alpha = self._load_alpha()
        self.theta = 6

    # -- lazy resources ---------------------------------------------------
    def _load_alpha(self) -> float:
        p = self.cfg.path("results") / "metrics.json"
        if p.exists():
            try:
                return float(util.read_json(p).get("best_alpha", 0.5))
            except Exception:
                return 0.5
        return 0.5

    def _gru_model(self) -> GRUScorer:
        if self._gru is None:
            self._gru = GRUScorer(str(self.cfg.path("gru") / "0"))
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
        return self._items_df

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
        items = self._items().set_index("item_id")
        out = []
        for iid in item_ids:
            if iid in items.index:
                row = items.loc[iid]
                out.append({"item_id": iid, "title": str(row["title"]), "text": str(row["text"])})
            else:
                out.append({"item_id": iid, "title": "", "text": ""})
        return out

    def _rank(self, user_id: str, history_df: pd.DataFrame, candidate_ids: list[str]) -> tuple[list[str], dict]:
        cand_items = self._candidate_items(candidate_ids)
        trace = asyncio.run(
            run_user_async(self.client, self.cfg, user_id, history_df, cand_items, self.theta)
        )
        hist_ids = history_df["item_id"].astype(str).tolist()
        gru_rank = self._gru_model().rank(hist_ids, candidate_ids)
        nli_scores = trace.nli.scores() if trace.nli else None
        fused = fuse(candidate_ids, trace.final_ranking, gru_rank, self.alpha, nli_scores)
        reasons = trace.reasons
        return fused, reasons

    # -- public API -------------------------------------------------------
    def preload(self) -> None:
        """Eagerly load heavy resources in the main thread before serving requests.

        Loading torch/sentence-transformers lazily inside a request thread can crash
        the process with an OMP pthread error.
        """
        self._items()
        for loader in (self._embedder_model, self._faiss, self._gru_model):
            try:
                loader()
            except Exception:
                pass

    def health(self) -> dict:
        gru_ok = (self.cfg.path("gru") / "0" / "model.pt").exists()
        faiss_ok = (self.cfg.path("faiss") / "index.faiss").exists()
        db_ok = Path(self.db_path).exists()
        return {
            "db": db_ok,
            "faiss": faiss_ok,
            "gru": gru_ok,
            "llm_key": self.cfg.llm_has_key(),
            "model": self.cfg.llm_model,
        }

    def list_users(self) -> list[dict]:
        conn = repo.connect(self.db_path)
        try:
            return repo.list_users(conn)
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
            fused, reasons = self._rank(user_id, history, candidate_ids)

            start = page * page_size
            page_ids = fused[start : start + page_size]
            items = self._render_items(page_ids, reasons)

            run_id = repo.create_run(
                conn, user_id, session_id, "feed",
                {"page": page, "page_size": page_size}, {"alpha": self.alpha},
            )
            repo.add_rec_items(conn, run_id, [
                {"item_id": iid, "rank": i, "score": 0.0, "reason": {}}
                for i, iid in enumerate(page_ids, start=1)
            ])
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
        import faiss
        ids = []
        # FAISS on user-context embedding
        titles = " ".join(history["title"].astype(str).tolist()[-20:])
        if titles.strip():
            emb = self._embedder_model().encode([titles], normalize_embeddings=True).astype(np.float32)
            idx = self._faiss()
            _, I = idx.search(emb, k)
            ids.extend(self._emb_ids[I[0]].tolist())
        # GRU top-k over in-vocab items
        gru = self._gru_model()
        hist_ids = history["item_id"].astype(str).tolist()
        if hist_ids:
            h_u = gru.user_vector(hist_ids)
            import torch
            with torch.no_grad():
                logits = h_u @ gru.model.embedding.weight.T  # vocab
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
            fused, reasons = self._rank(user_id, history, candidate_ids[:20])
            items = self._render_items(fused[:limit], reasons)
            run_id = repo.create_run(conn, user_id, session_id, "search",
                                     {"query": query, "filters": filters}, {"alpha": self.alpha})
            repo.add_rec_items(conn, run_id, [
                {"item_id": iid, "rank": i, "score": 0.0, "reason": {}}
                for i, iid in enumerate(fused[:limit], start=1)
            ])
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
        if q.get("max_price") is not None or (filters and filters.get("max_price")):
            mx = q.get("max_price") or (filters or {}).get("max_price")
            mask &= items["price"].fillna(float("inf")) <= mx
        for term in q.get("exclude_terms", []):
            mask &= ~items["text"].str.contains(term, case=False, na=False)
        eligible = items[mask]
        emb = self._embedder_model().encode([q.get("query", " ".join(q["keywords"]))], normalize_embeddings=True).astype(np.float32)
        idx = self._faiss()
        # search over all embeddings, then restrict to eligible ids
        _, I = idx.search(emb, k * 5)
        candidate_ids = [iid for iid in self._emb_ids[I[0]].tolist() if iid in set(eligible["item_id"].astype(str))]
        return self._filter_candidates(conn, user_id, candidate_ids, k)

    def _render_items(self, item_ids: list[str], reasons: dict) -> list[dict]:
        items = self._items().set_index("item_id")
        out = []
        for iid in item_ids:
            row = items.loc[iid] if iid in items.index else None
            title = str(row["title"]) if row is not None else iid
            text = str(row["text"]) if row is not None else ""
            cat = str(row["main_category"]) if row is not None else ""
            price = float(row["price"]) if row is not None and pd.notna(row.get("price")) else None
            out.append({
                "item_id": iid,
                "title": title,
                "category": cat,
                "price": price,
                "reason": reasons.get(iid, ""),
                "evidence": evidence_snippet(text),
                "score": 0.0,
            })
        return out

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

    def regenerate_profile(self, user_id: str) -> dict:
        conn = repo.connect(self.db_path)
        try:
            history = self._history_df(conn, user_id)
            from arag.agents.context import build_context
            from arag.agents.uua import run_uua
            c_st, c_lt = build_context(history, 5, 50)
            uua, _ = run_uua(self.client, self.cfg, c_lt, c_st)
            profile = uua.model_dump()
            event_ids = [str(e["event_id"]) for e in conn.execute(
                "SELECT event_id FROM interactions WHERE user_id=?", (user_id,)
            ).fetchall()]
            provenance = {"event_ids_used": event_ids, "model": self.cfg.llm_model}
            repo.save_profile(conn, user_id, profile, provenance, 1)
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
        # replay re-executes the request through the (cached) pipeline
        req = run["request_json"]
        if run["kind"] == "feed":
            return self.feed(run["user_id"], run["session_id"], req.get("page", 0), req.get("page_size", 10))
        if run["kind"] == "search":
            return self.search(run["user_id"], run["session_id"], req.get("query", ""), req.get("filters"), req.get("limit", 5))
        raise KeyError("unknown run kind")
