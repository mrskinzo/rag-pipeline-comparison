# rag-pipeline-comparison

A lightweight evaluation framework for experimenting with Retrieval-Augmented
Generation (RAG) pipelines against a real help‑center knowledge base.
Initially built to compare chunking strategies, overlaps and query
variants on GoDaddy’s documentation, the code can serve as a small
playground for anyone wanting to reproduce similar analyses.

---

## Features

* download and clean articles from a set of category pages (`scrape_articles.py`)
* inspect the raw data (`inspect_data.py`)
* build a vector store and try different RAG configurations
  (`build_pipelines.py`)
* perform systematic multi-question evaluations with scoring
  (`evaluate.py`)
* quick setup verification for external dependencies (`verify_setup.py`)

Scripts are now parameterised and importable, making them easier to test
and extend.

---

## Getting started

### 1. Clone the repository

```bash
git clone git@github.com:mrskinzo/rag-pipeline-comparison.git
cd rag-pipeline-comparison
```

### 2. Install dependencies

A minimal `requirements.txt` and `environment.yml` are provided for
reproducibility.  You can use conda or pip:

```bash
# using pip
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt

# or conda
conda env create -f environment.yml
conda activate rag-pipeline
```

> 💡 The environment file mirrors the GitHub Actions workflow; it only
> installs what the project actually needs rather than the entire
> container image.

### 3. Configure your API key

Create a `.env` in the project root containing:

```
ANTHROPIC_API_KEY=sk-xxx
```

The `verify_setup.py` script will warn you if the key is missing or if
external clients fail to initialise.

### 4. Run the workflow

Generate the articles JSON:

```bash
python scrape_articles.py --limit 50 --output data.json
```

Inspect what you collected:

```bash
python inspect_data.py --file data.json
```

Build and test individual pipelines:

```bash
python build_pipelines.py --question "How do I cancel a domain transfer?"
```

Run the full evaluation suite (uses the default 10 questions):

```bash
python evaluate.py --output results.csv
```

Results are written to CSV and printed as a summary table.  You can also
pass `--questions-file` to `evaluate.py` if you want to try your own
set of prompts.

---

## Development

* **Testing** – all reusable logic is covered by `pytest` tests under the
  `tests/` directory.  Run `pytest` to make sure your changes don’t
  regress anything.
* **Linting** – the GitHub Actions workflow already runs `flake8`; you can
  run the same locally with `pip install flake8`.
* **Formatting** – feel free to use `black`/`isort` or set up a
  pre-commit hook.

---

## Directory layout

```
├── articles.json          # scraped article dataset
├── build_pipelines.py     # ad‑hoc pipeline construction and demo
├── evaluate.py            # full evaluation harness with scoring
├── inspect_data.py        # tooling to examine articles.json
├── scrape_articles.py     # web crawler for GoDaddy help content
├── tests/                 # pytest suites
├── verify_setup.py        # checks environment/api clients
├── requirements.txt       # minimal pip deps
├── environment.yml        # conda environment
└── README.md              # this document
```

---

## Suggestions for further improvement

* package the code under a proper Python package (`src/…` layout)
* add type checking with `mypy` or Pyright and CI integration
* support alternative vector stores or LLM vendors via configuration
* provide a notebook demonstrating common analysis flows
* add caching to avoid re‑embedding when tweaking chunk sizes

Contributions and ideas welcome!

