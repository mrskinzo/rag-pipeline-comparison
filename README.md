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
  rag_core.py            → shared engine: chunking, cached embeddings,
         │                 Chroma vector stores, retrieval strategies,
         │                 generation, LLM-as-judge scoring
         ↓
  build_pipelines.py     → builds vector stores for each config preset
         ↓
  evaluate.py            → runs N questions through each pipeline, scores results
         ↓
  evaluation_results.csv → per-question, per-config scores
         ↓
  visualize_results.py   → summary tables + charts under reports/
```

### Configuration presets

Defined once in `rag_core.CONFIG_PRESETS` and shared by every script:

| Key | Label | Chunk Size | Overlap | k | Retrieval |
|---|---|---|---|---|---|
| `naive` | Config 1 - Naive | 256 | 32 | 3 | cosine |
| `optimized` | Config 2 - Optimized | 512 | 64 | 5 | cosine |
| `mmr` | Config 3 - MMR | 1024 | 128 | 5 | MMR |
| `multiquery` | Config 4 - MultiQuery | 1024 | 128 | 5 | multi-query |

**Retrieval strategies:**
- **Cosine similarity** — standard dense retrieval over `all-MiniLM-L6-v2` embeddings
- **MMR (Maximal Marginal Relevance)** — fetches extra candidates, then re-ranks to balance relevance against redundancy (`fetch_k` candidates, λ diversity trade-off)
- **Multi-query** — Claude generates 2 rephrasings of the question; results are merged and deduplicated

**Infrastructure details:**
- Chunk IDs are `sha1(url)`-based, so they're deterministic across processes
- Embeddings are cached on disk under `.cache/embeddings/` (keyed by model + content hash), so re-runs skip re-embedding unchanged text
- Chroma persists to `chroma_db/` by default; pass `--no-persist` for in-memory
- Large corpora are inserted into Chroma in batches of 500

---

## Evaluation methodology

Each configuration is scored on a fixed set of questions (LLM-as-judge with `claude-3-haiku-20240307`, no RAGAS dependency):

- **Faithfulness** — does the answer stay grounded in the retrieved chunks?
- **Answer relevancy** — does the answer address what was asked?
- **Context precision** — are the retrieved chunks actually relevant to the question?
- **Answer correctness** — is the answer factually correct, judged against the retrieved reference material?
- **Latency** — end-to-end retrieval + generation time

Judge calls that fail return a neutral 0.5, and all scores are clamped to [0, 1].

---

## Getting Started

```bash
git clone https://github.com/mrskinzo/rag-pipeline-comparison.git
cd rag-pipeline-comparison

# pip
python -m venv .venv && source .venv/bin/activate
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
python scrape_articles.py --limit 50 --output articles.json
python inspect_data.py --file articles.json
python build_pipelines.py --configs naive optimized mmr multiquery
python evaluate.py --configs naive optimized mmr --limit 5
python visualize_results.py
```

Useful flags:

```bash
python evaluate.py --configs naive mmr        # subset of presets
python evaluate.py --limit 3                  # first N questions only
python evaluate.py --questions-file q.txt     # custom questions, one per line
python evaluate.py --output my_results.csv    # custom output path
python evaluate.py --no-persist               # in-memory Chroma
```

Or via make: `make setup scrape inspect build eval viz test verify clean`

---

## Repository Structure

```
├── rag_core.py           # Shared core: chunking, embeddings, retrieval, scoring
├── scrape_articles.py    # Web crawler for GoDaddy help content
├── inspect_data.py       # Exploratory analysis of articles.json
├── build_pipelines.py    # Vector-store construction across config presets
├── evaluate.py           # Evaluation harness (CLI)
├── visualize_results.py  # Summary tables + charts from results CSV
├── verify_setup.py       # Checks environment and API clients
├── tests/                # pytest suites (run without an API key)
├── requirements.txt      # pip dependencies
├── environment.yml       # Conda environment mirror of CI
└── articles.json         # Scraped article dataset
```

---

## Development

```bash
pytest          # run tests (no API key needed)
flake8 .        # lint (mirrors GitHub Actions CI)
```

---

## What's Next

- [x] Caching layer to avoid re-embedding when only chunk config changes
- [x] MMR retrieval strategy
- [x] Visualisation of results (`visualize_results.py` → `reports/`)
- [x] Answer-correctness metric (LLM-as-judge)
- [ ] Notebook for interactive analysis
- [ ] Support for additional vector stores (Pinecone, Weaviate)
- [ ] Type checking with mypy
