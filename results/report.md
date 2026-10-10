# ARAG + GRU — Results Report

**Model:** gpt-4o-mini (openai)  
**Category:** Video_Games  
**θ*:** 5  **α*:** 0.1  
**LLM cost:** $0.9853  **Last run elapsed:** 2.3 min

## Main results (test users)

| System | hit@5 | ndcg@5 | hit@10 | ndcg@10 | mrr |
|---|---|---|---|---|---|
| random | 0.251 | 0.149 | 0.502 | 0.229 | 0.181 |
| popularity | 0.969 | 0.801 | 0.996 | 0.810 | 0.749 |
| gru_seed0 | 0.975 | 0.822 | 0.994 | 0.828 | 0.773 |
| recency | 0.490 | 0.389 | 0.670 | 0.447 | 0.403 |
| vanilla_rag | 0.480 | 0.386 | 0.670 | 0.448 | 0.403 |
| arag | 0.570 | 0.453 | 0.730 | 0.504 | 0.452 |
| arag_wo_nli | 0.610 | 0.481 | 0.720 | 0.516 | 0.472 |
| arag_wo_nli_csa | 0.590 | 0.456 | 0.690 | 0.488 | 0.445 |
| arag_gru | 0.990 | 0.802 | 1.000 | 0.805 | 0.740 |

## Conclusions (computed)

- ARAG+GRU vs ARAG (H1): Δ NDCG@5 = +0.349 (95% CI [+0.257, +0.441]) — significant
- ARAG+GRU vs GRU (H2): Δ NDCG@5 = +0.002 (95% CI [-0.018, +0.022]) — not significant
- LLM-Recency vs ARAG: Δ NDCG@5 = -0.064 (95% CI [-0.152, +0.022]) — not significant
- LLM-Vanilla-RAG vs ARAG: Δ NDCG@5 = -0.068 (95% CI [-0.151, +0.014]) — not significant
- ARAG w/o NLI vs ARAG: Δ NDCG@5 = +0.027 (95% CI [-0.025, +0.084]) — not significant
- ARAG w/o NLI&CSA vs ARAG: Δ NDCG@5 = +0.003 (95% CI [-0.045, +0.053]) — not significant

## Limitations

- Leave-last-out means other users' later interactions can appear in GRU training.
- 100 test users ⇒ wide CIs (roughly ±10 points on Hit@5).
- Uniform negatives are easier than real retrieval; not an exact paper reproduction.
- OpenAI gpt-4o-mini was used; results are ARAG-inspired, not a paper reproduction.

## Protocol

Frozen 20-item candidate lists (target + 19 uniform negatives) shared by all systems. 
θ tuned on validation NDCG@5 over {4,5,6,7}; α tuned on validation NDCG@5 over [0,1]. 
Non-LLM systems (Random, Popularity, GRU) reported on all test users; LLM systems on 100 sampled test users. 
Paired-bootstrap 95% CIs (10,000 resamples).
