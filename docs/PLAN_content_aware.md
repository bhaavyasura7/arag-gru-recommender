# Plan — branch `feat/content-aware` (zero new LLM spend)

Context for a fresh session: read this file, `README.md`, and `results/tables.md`. Do not re-read
the full `docs/REPORT.md` unless editing it.

**Motivation.** Median train history is 4 items (75% ≤ 7), so the ID-only GRU has little sequence
signal (all-user NDCG@5 0.279). A zero-training MiniLM content-similarity baseline reaches ≈0.41
on all 94,762 test users — the signal is in item text. Goal: a content-aware trained model, and
honest re-benchmarking. Every LLM output is cached; run the eval with `REPLAY_ONLY=1` so any
uncached call fails instead of costing money.

## Steps

1. **Done — baselines.** `src/arag/ranking/content.py` (`content_ranking`, `ItemKNN`), tests in
   `tests/test_content.py`, wired into `src/arag/eval/run.py` as systems `content`, `itemknn`
   (all test users) and `arag_content` (ARAG fused with content, α tuned on validation).
   Recency decay for both tuned on all validation users (`content.decay_grid`).
   Replay run: $0, 227 s, `make verify` OK, existing systems' metrics unchanged.
   Result (all 94,762 test users, NDCG@5): content 0.411, itemknn 0.406, gru_seed0 0.287.
   100-user table: content 0.439 vs ARAG 0.424 (n.s.); arag_content 0.423 (n.s. vs ARAG).
   Run the eval as `REPLAY_ONLY=1 OPENAI_API_KEY=replay-only-dummy make eval` (key never used).
2. **Done — text-aware GRU.** `GRU4Rec(text_feats=...)`: item vector = ID embedding +
   linear projection of the frozen MiniLM vector (tied input/output). `make gru-text` trains
   3 seeds at the best ID-only config (hidden 128, dropout 0.2, same 12-epoch cap) into
   `artifacts/gru_text/` (~5 min/seed). Training fixes: logits computed only on real
   (non-padded) positions and validation encoded in chunks capped at seq_len — both were
   needed to avoid OOM on 16 GB; the loss and the ID-GRU's recorded val score are unchanged.
   Val NDCG@5 0.476 / 0.473 / 0.477 (ID-only GRU 0.293). Still rising at the epoch cap.
3. **Done — re-benchmark** ($0 replay, `make verify` OK, original numbers unchanged).
   All 94,762 test users NDCG@5: gru_text 0.461/0.459/0.463 vs gru 0.287/0.275/0.275,
   content 0.411, itemknn 0.406. 100 users: gru_text_seed0 0.462, arag_gru_text 0.468
   (α*=0.3), arag 0.424. Text-GRU vs GRU +0.224 (sig); vs ARAG +0.038 (n.s.);
   ARAG+text-GRU vs ARAG +0.044 (n.s.), vs text-GRU +0.006 (n.s.).
4. **Docs.** Targeted edits only: REPORT.md abstract, §5.1 systems, §6 tables, §7 new analysis
   subsection, §12 conclusion, Appendix B; regenerate `docs/figures` via
   `scripts/make_figures.py`; new one-page `docs/OVERVIEW.md` (whole project at a glance).
