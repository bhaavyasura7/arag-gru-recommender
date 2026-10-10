# ARAG + GRU

An ARAG-style agentic recommender on Amazon Reviews 2023 (Video Games), extended with a
trained PyTorch **GRU sequential user model** fused into the final LLM ranking, plus a
React web demo with personalised feed, search, feedback, and persistent memory.

Full specification: [`SPEC.md`](SPEC.md).
**Project report (research write-up, methodology, results, threats to validity): [`docs/REPORT.md`](docs/REPORT.md).**
Detailed end-to-end process & training documentation with real run numbers: [`docs/PROCESS.md`](docs/PROCESS.md).

## Overview

- **Primary question:** on identical frozen candidate lists, does fusing a GRU score into
  ARAG's LLM ranking improve Hit@5 / NDCG@5 over ARAG alone and over the GRU alone?
- **Systems compared:** Random, Popularity, GRU, LLM-Recency, LLM-Vanilla-RAG, ARAG,
  two ARAG ablations, and **ARAG + GRU** (late fusion).
- **LLM:** configurable provider. The committed results were produced with OpenAI **`gpt-4o-mini`**
  (~$0.99 for the full evaluation). Local **`qwen3:8b`** via Ollama is also supported and is what
  the test-suite uses (zero cost, offline).
- **Embeddings:** sentence-transformers `all-MiniLM-L6-v2`, FAISS `IndexFlatIP`.

## Environment prerequisites

- Python 3.11+ (managed via [`uv`](https://github.com/astral-sh/uv)).
- Node.js + npm (for the frontend).
- **For the evaluation:** `OPENAI_API_KEY` in the environment when `llm.provider: openai`
  (the committed results used `gpt-4o-mini`), **or** a running Ollama server with the model pulled:
  ```bash
  ollama pull qwen3:8b
  ```
- **For `make test`:** a running Ollama server (`qwen3:8b`); tests always use the local provider.
- Network access to `huggingface.co` / `us.aws.cdn.hf.co` (dataset + MiniLM model) and PyPI/npm.
- Secrets are read from the environment only (`OPENAI_API_KEY`); never commit them.

## Setup

```bash
make setup       # venv + pip install + npm install
make freeze      # pin exact versions into requirements.txt (optional)
```

## Pipeline

```bash
make data        # download + manifest + parquet tables
make lists       # sequences, user samples, frozen 20-item candidate lists
make index       # MiniLM embeddings + FAISS
make gru         # train GRU (grid + 3 seeds)
make eval        # all systems on validation then test (resumable via cache)
make report      # results/*.md|json|png from artifacts
make all         # everything above
```

`make eval` prints the running cost and stops above `budget.max_usd` (inert for the local model).
Everything is resumable: re-running skips finished work and hits the LLM disk cache.

## Verification (no fabricated numbers)

```bash
make verify      # recompute metrics from raw predictions, diff against results/, check hashes
make test        # pytest (requires a running Ollama server; no API key)
```

`results/` is written only by the pipeline. Every metric is traceable to a real run via
`results/run_record.json` (git commit, config hash, seeds, model tag + digest, dataset manifest).
No stub or synthetic numbers are ever committed.

## Demo

```bash
make app         # build frontend, seed DB, run API on 127.0.0.1:8000
```

Open http://127.0.0.1:8000. `REPLAY_ONLY=1 make app` runs the demo offline, refusing any
uncached LLM call.

## Cost

The committed evaluation used OpenAI `gpt-4o-mini`: **$0.9853** for 2,405 calls
(4.89M input / 0.42M output tokens). Re-running with the LLM cache makes zero new API calls
($0). Local `qwen3:8b` is also supported at $0 but is ~10× slower per call on this hardware.

## How to read the results

- `results/tables.md` — main comparison table (100 test users, all systems, paired-bootstrap CIs).
- `results/alpha_curve.png` — validation/test NDCG@5 across α.
- `results/cost.json` — calls, tokens, USD, latency, parser repair rate, fallback counts.
- `results/report.md` — generated summary; conclusions are computed from the numbers.
