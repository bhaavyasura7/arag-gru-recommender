#!/usr/bin/env bash
# Run the pipeline for one Amazon category with a single, human-readable progress log.
#
#   scripts/run_category.sh <Category> nonllm   # phase A: data, lists, index, both GRUs, non-LLM eval
#   scripts/run_category.sh <Category> llm      # phase B: ARAG + LLM baselines (Ollama), report, verify
#
# Watch progress live:   tail -f logs/<Category>.log
# Every step is resumable: re-running skips finished work (downloads, LLM cache, saved models).
# Outputs go only to data/<Category>, artifacts/<Category>, results/categories/<Category>.
set -uo pipefail

CAT="${1:?usage: scripts/run_category.sh <Category> nonllm|llm}"
PHASE="${2:?usage: scripts/run_category.sh <Category> nonllm|llm}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
CFG="configs/categories/${CAT}.yaml"
LOG="logs/${CAT}.log"
mkdir -p logs
[ -f "$CFG" ] || "$PY" scripts/make_category_config.py "$CAT" >/dev/null

export ARAG_CONFIG="$CFG" PYTHONUNBUFFERED=1 TQDM_MININTERVAL=30

stamp() { date "+%Y-%m-%d %H:%M:%S"; }
say() { echo "[$(stamp)] $*" | tee -a "$LOG"; }

case "$PHASE" in
  nonllm) STEPS=(data lists index gru gru-text eval-nonllm) ;;
  llm)    STEPS=(eval report verify) ;;
  *) echo "phase must be nonllm or llm"; exit 2 ;;
esac

if [ "$PHASE" = llm ] && ! curl -s localhost:11434/api/tags >/dev/null; then
  say "Ollama is not running — start it with: ollama serve"; exit 1
fi

say "=================================================================="
say "[$CAT] phase=$PHASE  steps: ${STEPS[*]}"
say "config: $CFG   priority: nice $(ps -o nice= -p $$ | tr -d ' ')  (0 = full speed; run from your own Terminal)"
T0=$(date +%s)
i=0
for step in "${STEPS[@]}"; do
  i=$((i + 1))
  done_marker=""
  case "$step" in
    data)  done_marker="data/${CAT}/processed/items.parquet" ;;
    lists) done_marker="artifacts/${CAT}/lists/test_lists.parquet" ;;
    index) done_marker="artifacts/${CAT}/faiss/index.faiss" ;;
  esac
  if [ -n "$done_marker" ] && [ -f "$done_marker" ]; then
    say "[$CAT] step $i/${#STEPS[@]} $step — already done, skipping"; continue
  fi
  say "[$CAT] step $i/${#STEPS[@]} $step — started"
  S=$(date +%s)
  # caffeinate keeps the Mac awake while the step runs
  caffeinate -i make "$step" >>"$LOG" 2>&1
  rc=$?
  M=$(( ($(date +%s) - S) / 60 ))
  if [ $rc -ne 0 ]; then
    say "[$CAT] step $i/${#STEPS[@]} $step — FAILED (exit $rc) after ${M} min; see the lines above"
    exit $rc
  fi
  say "[$CAT] step $i/${#STEPS[@]} $step — done in ${M} min"
done
say "[$CAT] phase=$PHASE ALL DONE in $(( ($(date +%s) - T0) / 60 )) min"
