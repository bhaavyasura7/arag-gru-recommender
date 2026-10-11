# ARAG + GRU — Results Report

**Model:** gpt-4o-mini (openai)  
**Category:** Video_Games  
**θ*:** 5  **α*:** 0.9  
**LLM cost:** $0.0000  **Last run elapsed:** 2.3 min

## Main results (100 sampled test users, all systems)

| System | hit@5 | ndcg@5 | hit@10 | ndcg@10 | mrr |
|---|---|---|---|---|---|
| random | 0.220 | 0.125 | 0.470 | 0.205 | 0.162 |
| popularity | 0.200 | 0.109 | 0.470 | 0.197 | 0.150 |
| gru_seed0 | 0.390 | 0.238 | 0.640 | 0.321 | 0.249 |
| recency | 0.400 | 0.285 | 0.630 | 0.358 | 0.300 |
| vanilla_rag | 0.380 | 0.263 | 0.670 | 0.355 | 0.283 |
| arag | 0.530 | 0.424 | 0.700 | 0.479 | 0.432 |
| arag_wo_nli | 0.510 | 0.376 | 0.720 | 0.445 | 0.380 |
| arag_wo_nli_csa | 0.470 | 0.362 | 0.630 | 0.414 | 0.372 |
| arag_gru | 0.540 | 0.412 | 0.710 | 0.466 | 0.411 |

## Non-LLM systems on all 94762 test users

| System | hit@5 | ndcg@5 | hit@10 | ndcg@10 | mrr |
|---|---|---|---|---|---|
| random | 0.252 | 0.149 | 0.501 | 0.228 | 0.181 |
| popularity | 0.213 | 0.128 | 0.438 | 0.200 | 0.166 |
| gru_seed0 | 0.434 | 0.287 | 0.651 | 0.357 | 0.291 |

## Conclusions (computed)

- ARAG+GRU vs ARAG (H1): Δ NDCG@5 = -0.013 (95% CI [-0.032, +0.005]) — not significant
- ARAG+GRU vs GRU (H2): Δ NDCG@5 = +0.174 (95% CI [+0.069, +0.278]) — significant
- LLM-Recency vs ARAG: Δ NDCG@5 = -0.139 (95% CI [-0.217, -0.063]) — significant
- LLM-Vanilla-RAG vs ARAG: Δ NDCG@5 = -0.162 (95% CI [-0.247, -0.079]) — significant
- ARAG w/o NLI vs ARAG: Δ NDCG@5 = -0.049 (95% CI [-0.111, +0.011]) — not significant
- ARAG w/o NLI&CSA vs ARAG: Δ NDCG@5 = -0.062 (95% CI [-0.113, -0.016]) — significant

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
