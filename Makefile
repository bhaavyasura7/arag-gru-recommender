VENV := .venv
PY := $(VENV)/bin/python

.PHONY: setup freeze data lists index gru gru-text eval report test verify app all clean

## Create the venv and install Python + frontend dependencies.
setup:
	uv venv --python 3.11 $(VENV)
	uv pip install --python $(PY) -r requirements.txt
	cd frontend && npm install

## Pin exact installed versions (overwrites requirements.txt).
freeze:
	uv pip freeze --python $(PY) > requirements.txt

## Download raw data + metadata, write the manifest, build parquet tables.
data:
	$(PY) -m arag.dataprep.download
	$(PY) -m arag.dataprep.load

## Build sequences, user samples, and frozen candidate lists.
lists:
	$(PY) -m arag.dataprep.lists

## Embed item text and build the FAISS index.
index:
	$(PY) -m arag.retrieval.embed
	$(PY) -m arag.retrieval.index

## Train the GRU (grid + 3 seeds) and save checkpoints.
gru:
	$(PY) -m arag.models.train_gru

## Train the text-aware GRU (ID + MiniLM item vectors).
gru-text:
	$(PY) -m arag.models.train_gru --text

## Run all systems on validation (theta/alpha tuning) then test. Resumable via cache.
eval:
	$(PY) -m arag.eval.run

## Generate results/*.md|json|png from artifacts.
report:
	$(PY) -m arag.eval.report

## Audit: recompute metrics from raw predictions and diff against results/.
verify:
	$(PY) -m arag.eval.verify

## Run the test suite (requires a local Ollama server on 127.0.0.1:11434).
test:
	$(PY) -m pytest -q

## Build the frontend, seed the DB, run the API on 127.0.0.1:8000.
app:
	cd frontend && npm run build
	test -f artifacts/item_extras.parquet || $(PY) scripts/build_item_extras.py
	$(PY) -m arag.api.app

## Full pipeline: data -> lists -> index -> gru -> eval -> report.
all: data lists index gru gru-text eval report

clean:
	rm -rf $(VENV) artifacts data results/predictions \
		results/metrics.json results/tables.md results/alpha_curve.png \
		results/cost.json results/run_record.json results/report.md
