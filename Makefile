.PHONY: setup gen-data eval eval-fast demo test lint neo4j-up neo4j-down clean
PY ?= python3

setup:                ## install runtime + dev deps and the package (editable)
	$(PY) -m pip install -e ".[dev]"

gen-data:             ## (re)generate the synthetic corpus, graph, and gold set
	$(PY) data/generate_corpus.py

eval:                 ## full ablation (incl. cross-encoder rerank)
	$(PY) eval/run_eval.py

eval-fast:            ## ablation without the rerank configs (no cross-encoder download)
	$(PY) eval/run_eval.py --fast

demo:                 ## answer one question end-to-end with citations
	$(PY) scripts/demo.py "Who was the maternal grandfather of Nancy Ainsworth?"

test:                 ## run the test suite
	$(PY) -m pytest -q

lint:                 ## static checks
	$(PY) -m ruff check src eval data tests

neo4j-up:             ## start a local Neo4j and ingest the graph
	docker compose up -d neo4j
	@echo "waiting for neo4j..."; sleep 25
	NEO4J_URI=bolt://localhost:7687 $(PY) scripts/build_graph.py --load-neo4j

neo4j-down:
	docker compose down

clean:
	rm -rf .cache .pytest_cache **/__pycache__
