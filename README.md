# RAG Pipeline Comparison

An evaluation framework for systematically comparing RAG (Retrieval-Augmented Generation) pipeline configurations — chunk sizes, overlap strategies, and retrieval methods — against a real-world SaaS help center knowledge base.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python) ![License](https://img.shields.io/badge/License-MIT-green) ![Tests](https://img.shields.io/badge/Tests-pytest-orange)

---

## The Problem

RAG pipeline performance is highly sensitive to configuration choices — chunk size, overlap, retrieval strategy — but most teams pick these values by intuition rather than evidence. This project builds a lightweight, reproducible evaluation harness to answer: *which configuration actually retrieves the right chunks for real user questions?*

The knowledge base is GoDaddy's public help center documentation, chosen because it contains dense, domain-specific content that stress-tests retrieval systems.

---

## Architecture

```
[Raw Help Center URLs]
         ↓
  scrape_articles.py     → downloads and cleans article text into articles.json
         ↓
  inspect_data.py        → exploratory analysis of the scraped corpus
         ↓
  build_pipelines.py     → builds vector stores under different configurations
      ┌────────────────────────────────────────────────────────────────┐
      │  Config A: small non-overlapping chunks (500/0)                 │
      │            hybrid retrieval — dense (semantic) + BM25, RRF-fused │
      │  Config B: larger overlapping chunks (1000/200), semantic only  │
      │  Config C: small chunks (500/0), multi-query rewrite            │
      │            + cross-encoder re-ranker                            │
      └────────────────────────────────────────────────────────────────┘
         ↓
  evaluate.py            → runs N questions through each pipeline, scores results
         ↓
  results.csv            → per-question, per-config scores for analysis
```

---

## Technical Approach

**Evaluation methodology** (custom RAGAS-equivalent):

Each configuration is scored on a fixed set of questions across three dimensions:
- **Faithfulness** — does the answer stay grounded in the retrieved chunks?
- **Context precision** — are the retrieved chunks actually relevant to the question?
- **Answer correctness** — does the response match the expected answer?

Scoring is implemented in `evaluate.py` without RAGAS as a dependency, keeping the framework portable and transparent.

**Retrieval strategies tested:**
- **Dense / semantic** — embedding similarity over a ChromaDB collection (`all-MiniLM-L6-v2`).
- **BM25** — sparse keyword ranking (`rank-bm25`), which catches exact technical terms and brand/jargon tokens the embedding model never learned to encode.
- **Hybrid** — dense + BM25 fused with Reciprocal Rank Fusion (RRF), no score normalisation required.
- **Multi-query rewrite** — the LLM paraphrases the question to widen the candidate pool.
- **Cross-encoder re-ranker** — `cross-encoder/ms-marco-MiniLM-L-6-v2` (ships with `sentence-transformers`) re-orders the pool by jointly scoring each (query, chunk) pair.

The three configurations (`build_pipelines.py`, `evaluate.py`):

| Config | Chunking | Retrieval |
|---|---|---|
| **A** | small, non-overlapping (500 / 0) | hybrid — dense + BM25, RRF-fused |
| **B** | larger, overlapping (1000 / 200) | semantic (dense) only |
| **C** | small, non-overlapping (500 / 0) | multi-query rewrite + cross-encoder re-ranker |

---

## Running the comparison

`evaluate.py` runs the default 10-question suite through all three configurations
and scores each on faithfulness, answer relevancy, context precision, and latency,
writing per-question results to `evaluation_results.csv` and a mean summary to stdout.

The central question the harness is built to answer: **does single-pass hybrid
retrieval (Config A) actually beat dense-only (Config B) and the heavier
multi-query + re-ranker pipeline (Config C) on this corpus?** The honest answer
is whatever the numbers say — the framework exists precisely so the configuration
is chosen by evidence rather than intuition.

> **Reproduce the result, don't take a number on faith.** Running the suite
> requires an `ANTHROPIC_API_KEY` (generation + LLM-as-judge scoring) and network
> access to download the sentence-transformers embedder and cross-encoder. Run
> `python evaluate.py` and read the summary table it prints; that table is the
> finding. No winner is hard-coded into this README.

---

## Getting Started

```bash
git clone https://github.com/mrskinzo/rag-pipeline-comparison.git
cd rag-pipeline-comparison

# pip
python -m venv .env && source .env/bin/activate
pip install -r requirements.txt

# or conda
conda env create -f environment.yml && conda activate rag-pipeline
```

Create a `.env` file:

```
ANTHROPIC_API_KEY=sk-xxx
```

Run the full workflow:

```bash
python scrape_articles.py --limit 50 --output data.json
python inspect_data.py --file data.json
python build_pipelines.py --question "How do I cancel a domain transfer?"
python evaluate.py --output results.csv
```

---

## Repository Structure

```
├── scrape_articles.py    # Web crawler for GoDaddy help content
├── inspect_data.py       # Exploratory analysis of articles.json
├── build_pipelines.py    # Pipeline construction across configurations
├── evaluate.py           # Evaluation harness with scoring
├── verify_setup.py       # Checks environment and API clients
├── tests/                # pytest suites for core logic
├── requirements.txt      # Minimal pip dependencies
├── environment.yml       # Conda environment mirror of CI
└── articles.json         # Scraped article dataset
```

---

## Development

```bash
pytest          # run tests
flake8 .        # lint (mirrors GitHub Actions CI)
```

---

## What's Next

- [ ] Notebook for interactive analysis and visualisation of results
- [ ] Support for additional vector stores (Pinecone, Weaviate)
- [ ] Caching layer to avoid re-embedding when only chunk config changes
- [ ] Type checking with mypy
