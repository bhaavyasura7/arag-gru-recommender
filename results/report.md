# ARAG + GRU — Results Report

**Model:** gpt-4o-mini (openai)  
**Category:** Video_Games  
**θ*:** 5  **α*:** 0.2  
**LLM cost:** $1.1131 for all calls in this evaluation (paid in the last run: $0.0000; the rest came from the cache)  
**Last run elapsed:** 6.7 min

## Main results (100 sampled test users, all systems)

| System | hit@5 | ndcg@5 | hit@10 | ndcg@10 | mrr |
|---|---|---|---|---|---|
| random | 0.220 | 0.125 | 0.470 | 0.205 | 0.162 |
| popularity | 0.200 | 0.109 | 0.470 | 0.197 | 0.150 |
| content | 0.590 | 0.439 | 0.820 | 0.510 | 0.429 |
| itemknn | 0.540 | 0.399 | 0.670 | 0.442 | 0.392 |
| gru_seed0 | 0.680 | 0.519 | 0.850 | 0.575 | 0.499 |
| gru_text_seed0 | 0.720 | 0.506 | 0.900 | 0.565 | 0.465 |
| recency | 0.400 | 0.285 | 0.630 | 0.358 | 0.300 |
| vanilla_rag | 0.380 | 0.263 | 0.670 | 0.355 | 0.283 |
| arag | 0.530 | 0.424 | 0.700 | 0.479 | 0.432 |
| arag_wo_nli | 0.510 | 0.376 | 0.720 | 0.445 | 0.380 |
| arag_wo_nli_csa | 0.470 | 0.362 | 0.630 | 0.414 | 0.372 |
| arag_gru | 0.690 | 0.523 | 0.840 | 0.574 | 0.502 |
| arag_gru_text | 0.720 | 0.532 | 0.910 | 0.594 | 0.501 |
| arag_content | 0.590 | 0.423 | 0.810 | 0.494 | 0.411 |

## Non-LLM systems on all 94762 test users

| System | hit@5 | ndcg@5 | hit@10 | ndcg@10 | mrr |
|---|---|---|---|---|---|
| random | 0.252 | 0.149 | 0.501 | 0.228 | 0.181 |
| popularity | 0.213 | 0.128 | 0.438 | 0.200 | 0.166 |
| content | 0.569 | 0.411 | 0.771 | 0.476 | 0.402 |
| itemknn | 0.519 | 0.406 | 0.642 | 0.445 | 0.406 |
| gru_seed0 | 0.685 | 0.504 | 0.854 | 0.559 | 0.477 |
| gru_text_seed0 | 0.718 | 0.532 | 0.883 | 0.586 | 0.502 |

## Conclusions (computed)

- ARAG+GRU vs ARAG (H1): Δ NDCG@5 = +0.098 (95% CI [+0.013, +0.182]) — significant
- ARAG+GRU vs GRU (H2): Δ NDCG@5 = +0.004 (95% CI [-0.038, +0.045]) — not significant
- LLM-Recency vs ARAG: Δ NDCG@5 = -0.139 (95% CI [-0.217, -0.063]) — significant
- LLM-Vanilla-RAG vs ARAG: Δ NDCG@5 = -0.162 (95% CI [-0.247, -0.079]) — significant
- ARAG w/o NLI vs ARAG: Δ NDCG@5 = -0.049 (95% CI [-0.111, +0.011]) — not significant
- ARAG w/o NLI&CSA vs ARAG: Δ NDCG@5 = -0.062 (95% CI [-0.113, -0.016]) — significant
- Content similarity vs ARAG: Δ NDCG@5 = +0.014 (95% CI [-0.088, +0.116]) — not significant
- ItemKNN vs ARAG: Δ NDCG@5 = -0.025 (95% CI [-0.132, +0.082]) — not significant
- Text-aware GRU vs ID-only GRU: Δ NDCG@5 = -0.013 (95% CI [-0.064, +0.039]) — not significant
- Text-aware GRU vs ARAG: Δ NDCG@5 = +0.081 (95% CI [-0.016, +0.179]) — not significant
- ARAG+text-GRU vs ARAG (H1, text GRU): Δ NDCG@5 = +0.108 (95% CI [+0.016, +0.197]) — significant
- ARAG+text-GRU vs text-GRU (H2, text GRU): Δ NDCG@5 = +0.026 (95% CI [-0.001, +0.054]) — not significant
- ARAG+content vs ARAG: Δ NDCG@5 = -0.001 (95% CI [-0.070, +0.066]) — not significant

## Limitations

- Leave-last-out means other users' later interactions can appear in GRU training.
- 100 test users ⇒ wide CIs (roughly ±10 points on Hit@5).
- Sampled negatives are easier than real retrieval candidates; not an exact paper reproduction.
- OpenAI gpt-4o-mini was used; results are ARAG-inspired, not a paper reproduction.

## Protocol

Frozen 20-item candidate lists (target + 19 negatives) shared by all systems. 
Negatives: sampled from items with train interactions, proportional to train interaction count, excluding anything the user interacted with. 
θ tuned on validation NDCG@5 over {4,5,6,7}; α tuned on validation NDCG@5 over [0,1]. 
The main table and all significance tests use the same sampled test users for every system; non-LLM systems are additionally reported on all test users. 
Paired-bootstrap 95% CIs (10,000 resamples).
