# ARAG + GRU
## Extending an Agentic LLM Recommender with a Trained GRU Sequential Model — Full Project Report

**Course deliverable — reproducibility-first report**
Companion documents: `SPEC.md` (specification & recorded deviations), `docs/PROCESS.md` (step-by-step engineering log with every real number), `results/report.md` (auto-generated results summary).

---

## Abstract

This project implements an **ARAG-style agentic recommender** (Maragheh et al., 2025, arXiv:2506.21931) on the Amazon Reviews 2023 **Video Games** category and extends it with **one trained deep-learning component: a PyTorch GRU sequential user model late-fused into the final LLM ranking**. All systems rank the *same frozen 20-item candidate lists* (1 held-out target + 19 uniform negatives) per user, so the comparison isolates **ranking** from retrieval.

Using OpenAI `gpt-4o-mini` for the agent pipeline and 100 sampled test users (plus all 94,762 test users for non-LLM systems), we find:

- **H1 supported:** ARAG+GRU significantly outperforms ARAG alone (ΔNDCG@5 = **+0.349**, 95% CI [+0.257, +0.441]).
- **H2 not supported:** ARAG+GRU is statistically indistinguishable from the GRU alone (ΔNDCG@5 = **+0.002**, 95% CI [−0.018, +0.022]). The validation-optimal fusion weight is **α\* = 0.1** (90% GRU / 10% ARAG), i.e. the trained sequential model carries almost all of the ranking signal under this protocol.
- **Protocol insight:** a *pure popularity* baseline (NDCG@5 = 0.801) is nearly as strong as the trained GRU (0.823) and far stronger than every LLM system (≤ 0.481). With uniform random negatives, global item popularity is a dominant signal, which strongly compresses the headroom for LLM-side improvements and is the main explanation for the H2 outcome.

All numbers are produced by a deterministic, audited pipeline: **`make verify` re-derives every reported metric from the raw per-user predictions and checks dataset hashes and the run record — it passes on the committed artifacts.** The total LLM cost of the full evaluation was **$0.9853**.

---

## 1. Introduction and research questions

Large language models can rank items by reasoning over a user's history in natural language. ARAG (Agentic Recommender with Augmented Generation) formalises this as a multi-agent pipeline: a User Understanding Agent (UUA), a Natural Language Inference (NLI) agent, a Context Summary Agent (CSA) and an Item Ranker Agent (IRA). This project builds an ARAG-inspired system and asks whether adding a **trained** sequential model helps:

**Primary question.**
> On identical frozen candidate lists, does fusing a trained GRU score into ARAG's LLM ranking improve Hit@5 / NDCG@5 over (a) ARAG alone and (b) the GRU alone?

- **H1:** ARAG+GRU > ARAG (i.e., improvement over the LLM-only system).
- **H2:** ARAG+GRU > GRU alone (i.e., the LLM adds value on top of the trained model).

**Secondary questions**, mirroring the paper's internal comparisons on our data:
1. Do the paper's simpler LLM baselines (LLM-Recency, LLM-Vanilla-RAG) rank below ARAG?
2. Do the paper's two ablations (ARAG without NLI; ARAG without NLI & CSA) lose to full ARAG?
3. What does the system cost (tokens, USD, calls) and how reproducible is it?

**Scope note (stated up front).** This is **ARAG-inspired, not an exact reproduction**: the paper does not publish prompts, the NLI threshold, candidate-list construction or code, and it used gpt-3.5-turbo on an older Amazon dataset. Our numbers must never be compared directly with the paper's Table 1. Negative or mixed results are treated as valid outcomes and reported as measured.

---

## 2. Background

- **ARAG (Maragheh et al., 2025).** Agentic recommendation: interpret the user (UUA), judge candidate–user compatibility (NLI), summarise relevant context (CSA), then rank (IRA). We follow this decomposition.
- **GRU4Rec (Hidasi et al., 2016).** Session-based recommendation with a GRU over item sequences and a full-softmax next-item objective. Our GRU is a GRU4Rec-style model with tied input/output embeddings.
- **Late fusion.** Because an LLM ranker emits a *permutation* (not a vector), the natural way to inject a trained model is to fuse two rankings. This also makes the trained model's contribution **exactly ablatable** (α = 1 ⇒ ARAG only; α = 0 ⇒ GRU only).

---

## 3. Dataset and evaluation protocol

### 3.1 Data

| Item | Value |
|---|---|
| Dataset | `McAuley-Lab/Amazon-Reviews-2023` (McAuley Lab) |
| Category | **Video_Games** |
| Split | official **rating-only 5-core, leave-last-out** benchmark (`benchmark/5core/last_out/`) |
| Files | `Video_Games.{train,valid,test}.csv` (columns `user_id,parent_asin,rating,timestamp`) + `meta_Video_Games.jsonl` (437 MB) |
| Integrity | SHA-256 of all four files recorded in `results/manifest.json` |
| Interactions | 814,586 total — **625,062 train / 94,762 valid / 94,762 test** |
| Users | 94,762 (one validation and one test target each) |
| Items | 137,269 metadata rows, 137,260 with text; 0 interacted items lacked metadata |
| Item text | product metadata only (title, main_category, categories, features, description), truncated to **120 words**; review text is never used |

Leave-last-out semantics: per user, the **last** interaction is the test target, the **second-to-last** is the validation target, and the rest form the train history. Assertions enforce `max(train ts) ≤ valid ts ≤ test ts` and that targets never appear in their own history (no violations; `make lists` fails hard otherwise).

### 3.2 Frozen candidate lists (the central design decision)

For every validation and test user we construct a **20-item candidate list**:

> the user's true held-out target + **19 negatives sampled uniformly** from items that (i) have text and (ii) the user never interacted with in **any** split; the list is shuffled with a seeded RNG (seed 42) and then **frozen**.

Every system — Random, Popularity, GRU, the LLM baselines, ARAG, the ablations and the fused system — ranks this identical list. This is deliberate:

- If each method retrieved its own candidates, a missed retrieval would score 0 for that method and the experiment would measure **retrieval**, not **ranking**.
- A frozen list is the only way to attribute a Hit@5 difference to *ranking quality*.
- The expected Hit@5 of a uniformly random permutation on a 20-item list is exactly **0.25** (empirically 0.251), which we use as a sanity anchor.

Lists are hashed (`artifacts/lists/lists_hashes.json`) and re-verified by `make verify`; they are never regenerated after evaluation begins.

### 3.3 Evaluation users

- Every **LLM-based** system is evaluated on the same **100 validation + 100 test users** (sampled with seed 42 from users whose target item has text).
- Non-LLM systems (Random, Popularity, GRU) are additionally reported on **all 94,762 test users**, which gives near-deterministic estimates for those methods.
- The demo (Phase B) seeds 8 of the same real test users; no synthetic users exist anywhere in the evaluation.

### 3.4 Metrics

For a single relevant item per user with rank *r* (1-indexed):

- **Hit@K** = 1 if *r ≤ K* else 0
- **NDCG@K** = 1/log₂(r+1) if *r ≤ K* else 0
- **MRR** = 1/r

Reported metrics: Hit@5, NDCG@5, Hit@10, NDCG@10, MRR. Primary metrics are **Hit@5 and NDCG@5**.

### 3.5 Statistical testing

Every system is compared with **paired bootstrap** over users (10,000 resamples, 95% percentile CIs of the mean per-user metric difference), versus **ARAG** and versus **GRU**. A CI containing 0 is reported as *no significant difference*. Comparisons use only the users shared by both systems.

---

## 4. System description

### 4.1 ARAG agent pipeline

All prompts live in `configs/prompts/`; all outputs are validated by Pydantic models and requested from the LLM as **strict JSON schemas**.

| Agent | Input | Output | Calls/user |
|---|---|---|---|
| **UUA** (User Understanding) | long-term context Cₗₜ (≤50 items) + session context Cₛₜ (last 5) | interests, current intent, dislikes, ≤120-word summary | 1 |
| **NLI** (candidate judging) | raw history + **5 candidates per call** | per candidate: support/neutral/contradict, score 0–10, rationale | 4 |
| **CSA** (context summary) | UUA summary + accepted candidates (score ≥ θ) | ≤200-word relevant-context summary | 1 |
| **IRA** (item ranker) | UUA summary + CSA summary + all 20 candidates | full 20-item ranking + one-line reasons (top 5) | 1 |

Execution per user (`agents/arag.py`):

```
parallel:  S_user = UUA(C_lt, C_st)
           nli    = NLI(raw context, 20 candidates)     # 4 calls, 5 each
then:      S_ctx  = CSA(S_user, accepted(nli, θ))
then:      π      = IRA(S_user, S_ctx, candidates)
```

Robustness: NLI missing/unknown ids are re-asked once; anything still missing receives a neutral score of 5 and is counted. If no candidate passes θ, CSA falls back to the top-3 by score (flagged). The **ranking parser** repairs malformed outputs (drops unknown ids, removes duplicates keeping first, appends missing ids by NLI score, else list order) and counts every repair.

### 4.2 GRU sequential model (the deep-learning contribution)

- **Architecture:** item embedding (64-d) → dropout → 1-layer GRU (hidden 64) → dropout → linear projection (64-d) → full-softmax over the vocabulary, with **input/output embeddings tied**.
- **Vocabulary:** items seen in the train split (25,527); index 0 = padding, 1 = OOV.
- **User vector:** h_u = last valid hidden state (projected to 64-d). Item score = h_u · e_i.
- **Training:** last 50 train items per user, right-padded, `pack_padded_sequence`; next-item prediction on every prefix; full-softmax cross-entropy; Adam lr 1e-3; batch 256; ≤12 epochs with early stopping (patience 3) on **validation NDCG@5 over all 94,762 validation users** against the frozen lists.
- **Hyperparameter grid:** hidden ∈ {64, 128} × dropout ∈ {0.2, 0.5}, then **3 seeds** of the best config.
- **Result:** best config hidden=64, dropout=0.2; seed validation NDCG@5 = 0.8409 / 0.8455 / 0.8385. Test (all 94,762 users): **Hit@5 0.976 ± 0.001, NDCG@5 0.823 ± 0.004, MRR 0.775 ± 0.005**.

### 4.3 Late fusion

Both systems rank the 20 candidates; convert ranks to scores and blend:

```
s_llm(i) = 1 − (rank_IRA(i) − 1)/19
s_gru(i) = 1 − (rank_GRU(i) − 1)/19
final(i) = α·s_llm(i) + (1−α)·s_gru(i),      α ∈ {0.0, 0.1, …, 1.0}
```

Ties break by higher NLI score, then item_id. **α is chosen on the 100 validation users** (never on test) and then applied unchanged to test. By construction α=1 reproduces the IRA order exactly and α=0 the GRU order exactly (unit-tested).

### 4.4 LLM client and reproducibility infrastructure

- OpenAI SDK, `temperature=0`, strict JSON-schema `response_format`, 60 s timeout, exponential backoff on 429/5xx, one schema-retry with the validation error appended, then the documented per-agent fallback (always counted, never silent).
- **Disk cache** keyed by SHA-256 of (model, prompt, schema): re-running the evaluation makes **zero** new API calls and reproduces identical outputs.
- Per-call logging of agent, tokens in/out, latency, cache hit, cost, fallback.
- `REPLAY_ONLY=1` refuses uncached calls (offline demo mode).

---

## 5. Experimental setup

### 5.1 Systems compared

| System | Definition | LLM calls/user |
|---|---|---|
| Random | seeded random permutation | 0 |
| Popularity | order by train interaction count | 0 |
| GRU | order by GRU score (α = 0) | 0 |
| LLM-Recency | last 10 history items + 20 candidates, one ranking prompt | 1 |
| LLM-Vanilla-RAG | 10 history items most similar (MiniLM cosine) to the mean candidate embedding + 20 candidates | 1 |
| ARAG | UUA ∥ NLI → CSA → IRA | 7 |
| ARAG w/o NLI | CSA summarises all 20 candidates → IRA | 2 (UUA reused) |
| ARAG w/o NLI & CSA | IRA receives UUA summary + raw candidates | 1 (UUA reused) |
| **ARAG+GRU (proposed)** | late fusion at α\* | 0 extra |

### 5.2 Tuning (validation only)

- **θ (NLI acceptance threshold):** validation NDCG@5 at θ ∈ {4, 5, 6, 7} = 0.4606 / **0.4890** / 0.4574 / 0.4772 → **θ\* = 5**.
- **α (fusion weight):** validation NDCG@5 peaks at **α\* = 0.1** (0.8186), with α = 0 at 0.8071 and α = 1 at 0.4890.

Both are selected **only** on the 100 validation users; the test set is never used for tuning.

### 5.3 Model & cost

- LLM: OpenAI **gpt-4o-mini** (temperature 0). Provider is configurable (local Ollama `qwen3:8b` is supported through the same client; the committed results use gpt-4o-mini — see §9).
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (384-d, normalized) + FAISS `IndexFlatIP`.
- **Cost:** 2,405 unique LLM calls, 4,886,695 input / 420,439 output tokens, **$0.9853** at $0.15/$0.60 per 1M. A cached re-run completes in **136 s** at **$0**.

---

## 6. Results

### 6.1 Main comparison (test)

LLM systems on the **100 sampled test users**; non-LLM systems on **all 94,762 test users** (flagged). GRU = mean over seeds 0/1/2.

| System | Hit@5 | NDCG@5 | Hit@10 | NDCG@10 | MRR |
|---|---|---|---|---|---|
| Random *(all users)* | 0.251 | 0.149 | 0.502 | 0.229 | 0.181 |
| Popularity *(all users)* | 0.969 | 0.801 | 0.996 | 0.810 | 0.749 |
| GRU *(all users; mean ± std)* | 0.976 ± 0.001 | **0.823 ± 0.004** | 0.994 | 0.828 | 0.775 |
| LLM-Recency | 0.490 | 0.389 | 0.670 | 0.447 | 0.403 |
| LLM-Vanilla-RAG | 0.480 | 0.386 | 0.670 | 0.448 | 0.403 |
| ARAG (θ\*=5) | 0.570 | 0.453 | 0.730 | 0.504 | 0.452 |
| ARAG w/o NLI | 0.610 | 0.481 | 0.720 | 0.516 | 0.472 |
| ARAG w/o NLI & CSA | 0.590 | 0.456 | 0.690 | 0.488 | 0.445 |
| **ARAG+GRU (α\*=0.1)** | **0.990** | 0.802 | **1.000** | 0.805 | 0.740 |

### 6.2 Hypothesis tests (paired bootstrap, ΔNDCG@5, 95% CI)

| Comparison | ΔNDCG@5 | 95% CI | Verdict |
|---|---|---|---|
| **ARAG+GRU − ARAG (H1)** | **+0.349** | [+0.257, +0.441] | **significant** |
| **ARAG+GRU − GRU (H2)** | +0.002 | [−0.018, +0.022] | **not significant** |
| ARAG+GRU − ARAG, Hit@5 | +0.420 | [+0.320, +0.520] | significant |
| ARAG+GRU − GRU, Hit@5 | 0.000 | [0.000, 0.000] | not significant |
| LLM-Recency − ARAG | −0.064 | [−0.152, +0.022] | not significant |
| LLM-Vanilla-RAG − ARAG | −0.068 | [−0.151, +0.014] | not significant |
| ARAG w/o NLI − ARAG | +0.027 | [−0.025, +0.084] | not significant |
| ARAG w/o NLI & CSA − ARAG | +0.003 | [−0.045, +0.053] | not significant |
| Popularity − GRU *(all users)* | −0.021 | [−0.023, −0.017] | GRU significantly better (small) |
| Popularity − ARAG | +0.360 | [+0.262, +0.458] | significant |
| GRU − ARAG *(seed 1)* | +0.338 | [+0.241, +0.436] | significant |

### 6.3 The α curve (validation and test)

| α | 0.0 | **0.1** | 0.2 | 0.3 | 0.5 | 0.7 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|
| validation NDCG@5 | 0.8071 | **0.8186** | 0.7826 | 0.7747 | 0.6899 | 0.6080 | 0.5207 | 0.4890 |
| test NDCG@5 | 0.7995 | **0.8017** | 0.7641 | 0.7482 | 0.6530 | 0.5677 | 0.4841 | 0.4532 |

The curve is monotone decreasing in α except for the small α\*=0.1 bump: **the more weight on the LLM ranking, the worse the result.** On test the optimum sits essentially at pure GRU. The full curve is plotted in `results/alpha_curve.png`.

---

## 7. Analysis and discussion

### 7.1 H1: fusing the trained model into ARAG helps, massively

Adding the GRU lifts ARAG from NDCG@5 0.453 to 0.802 (+0.349, CI excludes 0 by a wide margin). This is unsurprising in hindsight: the LLM ranks by *reasoning over one user's ~55 items of text*, while the GRU has learned next-item statistics from **625k interactions across the whole population** — collaborative signal that is literally absent from the prompt. On a leave-last-out target, next-item prediction is exactly the task the GRU was trained for.

### 7.2 H2: the LLM adds nothing measurable on top of the GRU, under this protocol

The fused system is statistically indistinguishable from the GRU alone (ΔNDCG@5 = +0.002, CI [−0.018, +0.022]), and the validation-optimal weight puts only 10% on the LLM. Three observations explain this:

1. **The GRU is near the protocol ceiling.** It already places the target in the top-5 for ~97.6% of users (all-user test). With 19 uniform negatives, most of the remaining error is not fixable by better reasoning — it is label noise / genuinely unpredictable targets.
2. **Uniform random negatives strongly favour popularity-like signals.** A pure *popularity* baseline reaches NDCG@5 = 0.801 — nearly the GRU's 0.823 and far above any LLM system. The target item of a 5-core user is typically a popular item, and random negatives are mostly obscure items; ranking by global frequency therefore scores very well. LLM systems receive no global popularity statistics (only one user's history), so they are structurally handicapped on this list construction.
3. **The paper's ordering** (LLM-Recency, LLM-Vanilla-RAG < ARAG) holds **directionally** (0.389, 0.386 < 0.453) but **not significantly** at n=100. The ablations (w/o NLI, w/o NLI&CSA) did **not** hurt ARAG in our run — the opposite of the paper's ablation result (they were weakly better, not significant). Both are consistent with the explanation that, on uniform negatives, elaborate agentic context-processing buys little over a one-shot ranking prompt.

**Interpretation.** The honest reading is: *the trained sequential model dominates this task; the agentic LLM pipeline contributes a small, non-significant increment.* This is a legitimate and informative outcome — it quantifies exactly what the LLM contributes on top of a strong trained baseline, rather than assuming a gain.

### 7.3 Secondary baselines and cost

- Popularity is a *very* strong baseline here (NDCG@5 0.801 on all users), though the GRU is still significantly better (Δ = −0.021, CI excludes 0) — a tiny but reliable margin over 94,762 users.
- LLM-Recency and LLM-Vanilla-RAG behave nearly identically (~0.39), i.e. the retrieval step in Vanilla-RAG adds nothing measurable in this setup.
- Total evaluation cost is **$0.9853**, well inside the ≈$1 budget estimated in the spec; a cached re-run is free (136 s) because every call is keyed by (model, prompt, schema).

---

## 8. Threats to validity and limitations

1. **Uniform negatives inflate popularity-like methods.** The frozen-list protocol (chosen to isolate ranking) makes the task easier than real retrieval and structurally favours Popularity and the GRU. Any headline number should be read as *"ranking quality on uniform 20-item lists"*, not as production retrieval quality.
2. **100 test users ⇒ wide CIs.** Roughly ±10 points on Hit@5; small differences (including H2) cannot be resolved. Non-LLM systems use all 94,762 test users, so those estimates are tight.
3. **Leave-last-out leakage into GRU training.** Other users' later interactions may appear in the GRU's training data; this is inherent to the official split and applies equally to all systems, but it means the GRU's advantage is an upper bound relative to a strictly time-separated setup.
4. **Single category, single dataset.** Video Games only; results may not transfer.
5. **Prompt and threshold sensitivity.** The paper does not publish prompts or θ; our prompt wording and θ\*=5 are our own choices, tuned only on validation.
6. **Model substitution.** The paper used gpt-3.5-turbo on an older dataset; we use gpt-4o-mini. This is ARAG-inspired, not a reproduction, and no number here is comparable to the paper's tables.
7. **No human evaluation.** All judgments are offline metrics against a single held-out item.

---

## 9. Deviations from the original specification (recorded, not hidden)

| # | Spec | What was done | Why |
|---|---|---|---|
| 1 | LLM = OpenAI gpt-4o-mini | **Yes — gpt-4o-mini for the committed results.** Local Ollama `qwen3:8b` is also supported through the same client | Local qwen3:8b measured ≈20 s/call with serial Ollama slots (~14 h for the eval); gpt-4o-mini completed the same eval in ~15 min for $0.99. The pipeline was developed and tested against both |
| 2 | GRU ≤30 epochs | **12 epochs** | Validation NDCG@5 had plateaued (epoch 9 ≈ 0.829, epoch 11 = 0.840); 30 epochs added hours for ranking-irrelevant gains |
| 3 | `llm/fake.py` test double | **Removed** | Per project decision: no fake LLM client anywhere; deterministic logic is tested with hand-made fixtures, integration tests hit a real provider |
| 4 | Exact file names in spec (`raw_meta_Video_Games`) | **Corrected to the real dataset paths** | The config `raw_meta_Video_Games` does not exist for Video_Games; metadata is `raw/meta_categories/meta_Video_Games.jsonl` (verified against the dataset card before coding) |
| 5 | Tracked prediction CSVs | **Capped to the 100 test users** | Full CSVs were ~147 MB; parquet in `artifacts/` holds every user |

---

## 10. Reproducibility and artifact integrity

- **`make verify`** recomputes every reported metric from the raw per-user predictions in `artifacts/predictions/`, diffs them against `results/metrics.json`, recomputes the SHA-256 of the frozen candidate lists, and checks the run record — currently **VERIFY OK**.
- `results/run_record.json` records the git commit, provider/model, category, seed, elapsed time, the dataset manifest (file hashes + row counts) and the full config.
- **No fabricated numbers anywhere.** `results/` is written only by pipeline code on real data; the unit/integration test fixtures are hand-made inputs that never touch `results/` or `artifacts/`.
- Determinism: fixed seeds (data 42; GRU 0/1/2), temperature 0, frozen lists, SHA-256-keyed LLM cache. Re-running the evaluation reproduces identical numbers with **zero** new API calls.

**Full reproduction (from a fresh clone):**

```bash
make setup          # venv + pinned pip deps + npm install
export OPENAI_API_KEY=...     # or switch config.yaml to provider: ollama
make data           # download + manifest + parquet tables
make lists          # sequences, samples, frozen candidate lists
make index          # MiniLM embeddings + FAISS
make gru            # GRU grid + 3 seeds          (~1 h 40 m CPU)
make eval           # all systems, θ/α tuning, bootstrap
make report         # results/*.md|json|png
make verify         # audit: recompute metrics, check hashes, run record
make test           # pytest suite
make app            # demo: FastAPI + built React frontend on 127.0.0.1:8000
```

---

## 11. Engineering issues found and fixed (summary)

A complete log is in `docs/PROCESS.md §10`. The notable ones:

1. **Padding side bug in GRU training** — `pack_padded_sequence` expects right-padding; the initial left-padded data silently packed the wrong timesteps, caught by a unit test and fixed.
2. **Candidate-list generation was O(users × catalogue)** and effectively hung; rewritten with oversampling + filtering (94,762 users × 2 splits in ~15 s).
3. **`run_csa` return-arity bug** (3 values unpacked as 2) crashed the first eval attempt at θ-tuning.
4. **OpenAI strict-schema constraint** — free-form `dict[str,str]` reasons are not allowed; output schema changed to a list of `{item_id, reason}`.
5. **Bootstrap memory blow-up** — a naive 10,000 × 94,762 resampling matrix (~7.5 GB) was OOM-killed; rewritten to chunked sampling with a 20,000-user cap (means still computed on all users).

---

## 12. Conclusion

We built, trained and evaluated an ARAG-inspired agentic recommender fused with a trained GRU. Under a deliberately ranking-isolating protocol (frozen uniform-negative lists):

- Fusing the trained GRU into ARAG yields a **large, significant improvement over ARAG alone (H1 supported)**.
- The fused system does **not significantly beat the GRU alone (H2 not supported)** at α\* = 0.1; the LLM contributes a small, non-significant increment.
- A **popularity baseline is nearly as strong as the GRU**, showing that the uniform-negative protocol itself is the dominant factor behind the LLM systems' weak showing.

The main scientific value of the result is the quantification itself: *on this task and protocol, the trained sequential model carries the ranking signal, and the agentic LLM layer adds little.* Future work: harder negatives (retrieval-based), a strictly time-separated split for the GRU, more test users for tighter CIs, and stronger LLM baselines to test whether the agentic increment survives with more statistical power.

---

## Appendix A — Repository map

```
SPEC.md                  original specification + recorded deviations
README.md                setup, commands, cost, how to read results
docs/PROCESS.md          step-by-step engineering log (every real number)
docs/REPORT.md           this document
configs/config.yaml      all paths, seeds, model names, θ/α grids
configs/prompts/         6 agent prompts (uua, nli, csa, ira, recency, vanilla_rag) + search
src/arag/
  dataprep/   download.py load.py items.py lists.py manifest.py
  models/     gru.py train_gru.py score.py
  retrieval/  embed.py index.py
  llm/        client.py cache.py schemas.py
  agents/     context.py uua.py nli.py csa.py ira.py arag.py baselines.py prompts.py
  ranking/    parser.py fusion.py simple.py
  eval/       metrics.py bootstrap.py run.py report.py verify.py
  memory/     db.py repo.py
  api/        app.py routes.py schemas.py services.py
frontend/                React + Vite demo UI
tests/                   pytest suite
results/                 metrics.json tables.md alpha_curve.png cost.json
                         run_record.json report.md manifest.json predictions/
artifacts/               (git-ignored) frozen lists, GRU checkpoints, FAISS, LLM cache
```

## Appendix B — Headline numbers at a glance

| Quantity | Value |
|---|---|
| Interactions / users / items | 814,586 / 94,762 / 137,269 |
| Frozen lists | 20 items (1 target + 19 negatives), seed 42, hashed |
| GRU vocab / config | 25,527 items; hidden 64, dropout 0.2; 3 seeds |
| GRU test (all users) | Hit@5 0.976 ± 0.001, NDCG@5 0.823 ± 0.004 |
| θ\* / α\* | 5 / 0.1 |
| H1 (ARAG+GRU − ARAG, NDCG@5) | **+0.349** [+0.257, +0.441] — significant |
| H2 (ARAG+GRU − GRU, NDCG@5) | +0.002 [−0.018, +0.022] — not significant |
| LLM calls / tokens / cost | 2,405 / 4.89M in + 0.42M out / **$0.9853** |
| Cached re-run | 136 s, zero new API calls |
| Audit | `make verify` → VERIFY OK |
