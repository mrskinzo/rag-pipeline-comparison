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
      ┌──────────────────────────────────────────┐
      │  Config A: chunk=256,  overlap=32,  cosine │
      │  Config B: chunk=512,  overlap=64,  cosine │
      │  Config C: chunk=1024, overlap=128, MMR    │
      └──────────────────────────────────────────┘
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
- Cosine similarity (standard dense retrieval)
- MMR (Maximal Marginal Relevance) — reduces redundancy in retrieved chunks

---

## Key Findings

Running the default 10-question evaluation suite against GoDaddy help content:

| Config | Chunk Size | Overlap | Retrieval | Avg Score |
|---|---|---|---|---|
| A | 256 | 32 | Cosine | baseline |
| B | 512 | 64 | Cosine | +12% vs A |
| C | 1024 | 128 | MMR | best on multi-part questions |

Larger chunks (512–1024) consistently outperformed 256-token chunks on complex questions. MMR improved diversity but slightly hurt precision on narrow factual queries.

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
