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
2. **Text-aware GRU.** In `src/arag/models/gru.py`, add an option to initialise/augment item
   embeddings with the MiniLM vectors (`artifacts/faiss/item_embeddings.npy`, 384-d → linear
   projection to 64-d, plus a learned ID embedding). Config flag `gru.text_init`. Train one
   config + 3 seeds in the background (`make gru`), ~15–20 min per run on CPU. Add as system
   `gru_text_seed{0,1,2}` alongside the existing GRU, and `arag_gru_text` fusion.
3. **Re-benchmark.** `REPLAY_ONLY=1 make eval && make report && make verify`; `make test`.
4. **Docs.** Targeted edits only: REPORT.md abstract, §5.1 systems, §6 tables, §7 new analysis
   subsection, §12 conclusion, Appendix B; regenerate `docs/figures` via
   `scripts/make_figures.py`; new one-page `docs/OVERVIEW.md` (whole project at a glance).
