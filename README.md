# RAG Pipeline Comparison

A small, reproducible harness for comparing RAG retrieval configurations against a
real help-center corpus, scored with an LLM judge.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python) ![License](https://img.shields.io/badge/License-MIT-green) ![Tests](https://img.shields.io/badge/Tests-pytest-orange)

The question: **does bigger-chunk retrieval actually beat naive small-chunk
retrieval on real support questions?** Most teams pick chunk size and `k` by
intuition. This runs the comparison and reports what happened, including the
parts that didn't go the way I expected.

---

## What it actually does

Dense retrieval only. Chroma with `all-MiniLM-L6-v2` embeddings, Claude Haiku for
generation, and a separate Haiku call as the judge.

```
scrape_articles.py   → GoDaddy help articles      → articles.json
inspect_data.py      → corpus stats
evaluate.py          → 10 questions × 3 configs   → evaluation_results.csv
```

Three configs, defined in `evaluate.py`:

| Config | Chunk | Overlap | k | Retrieval |
| --- | --- | --- | --- | --- |
| 1 — Naive | 500 | 0 | 3 | dense (cosine) |
| 2 — Optimized | 1000 | 200 | 5 | dense (cosine) |
| 3 — MultiQuery | 1000 | 200 | 5 | dense + 2 LLM rewrites, fused with Reciprocal Rank Fusion |

**There is no BM25, no hybrid retrieval, and no cross-encoder re-ranker here.**
If you're looking for those, this repo doesn't have them (see [Limitations](#limitations)).

## Metrics

Three scores per answer, each a 0–1 rating from an LLM judge (`evaluate.py`):

- **Faithfulness** — is every claim in the answer supported by the retrieved context?
- **Answer relevancy** — does the answer actually address the question?
- **Context precision** — were the retrieved chunks relevant? (mean over the k chunks)

Plus wall-clock latency per query.

---

## Results

10 questions × 3 configs, one run. Reproduce with `python evaluate.py`; raw
per-question rows are in [`evaluation_results.csv`](evaluation_results.csv).

| Config | Faithfulness | Answer relevancy | Context precision | Latency |
| --- | --- | --- | --- | --- |
| 1 — Naive (500/0, k=3) | **0.955** | 0.850 | **0.820** | 2.39s |
| 2 — Optimized (1000/200, k=5) | 0.940 | 0.865 | 0.522 | **2.15s** |
| 3 — MultiQuery (1000/200, k=5, RRF) | 0.890 | **0.915** | 0.531 | 3.05s |

**The naive config won on precision, and it wasn't close.** Small non-overlapping
chunks scored 0.820 on context precision against 0.522 for the larger overlapping
ones. The intuition that "bigger chunks with overlap retrieve better" did not hold
here. The likely reason is mundane: a 1000-char chunk pulls in more surrounding
text that has nothing to do with the question, and the judge marks it down. At
`k=5` you also retrieve five of them instead of three, so there's more room to be
wrong.

**Multi-query bought relevancy and paid for it in latency and faithfulness.** It
scored best on answer relevancy (0.915) but worst on faithfulness (0.890), and it
is 42% slower than Config 2 because each query costs an extra LLM call to generate
the rewrites. Widening retrieval surfaces more usable context and more
distraction at the same time.

**Nothing here is a large effect at n=10.** See [Limitations](#limitations) before
you take any of this to a design review.

### About the multi-query config

The rewrites are fused with [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf):
each chunk scores `sum(1 / (60 + rank))` across the ranked lists it appears in, so
a chunk that several rewrites agree on outranks one that a single rewrite loved.

This matters. An earlier version of this repo concatenated the rewrites' hits,
deduped, and truncated to `k` — but the original query alone already returns `k`
chunks, so every rewrite got sliced off before it was used. **Config 3 retrieved
byte-identical context to Config 2 on all 10 questions: it was a no-op that cost
an extra API call.** With RRF it now retrieves different context on 7 of 10.
`tests/test_retrieval.py` pins that behavior so it can't regress.

---

## Getting started

```bash
git clone https://github.com/mrskinzo/rag-pipeline-comparison.git
cd rag-pipeline-comparison

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# or: conda env create -f environment.yml && conda activate rag-pipeline
```

Add a `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Then:

```bash
python scrape_articles.py   # rebuild articles.json (optional — it's committed)
python inspect_data.py      # corpus stats
python evaluate.py          # the comparison; ~5-8 min, writes evaluation_results.csv
pytest                      # tests (no API key needed)
```

`scrape_articles.py` takes `--limit` (default 40), `--output`, and `--categories`.
`inspect_data.py` takes `--file`. **`evaluate.py` takes no arguments** — the
configs, questions, and output path are constants at the top of the file. Set
`EVAL_MODEL` to judge with something other than Haiku.

## Layout

```
scrape_articles.py      crawler for the GoDaddy help center
inspect_data.py         corpus stats
evaluate.py             chunking, retrieval, RRF fusion, generation, judging
verify_setup.py         checks env + API client
articles.json           the scraped corpus (committed, so the eval is reproducible)
evaluation_results.csv  per-question output of the last run
tests/                  pytest; no API key required
```

---

## Limitations

Read these before citing any number above.

- **n=10.** Ten questions, one run, no confidence intervals and no significance
  testing. Treat the table as directional, not conclusive.
- **The judge is Haiku grading Haiku.** Same model family generating and scoring,
  which is a known source of self-evaluation bias.
- **No ground-truth answers.** There's no gold answer per question, so
  "faithfulness" means grounded in the retrieved chunks, not *correct*. A pipeline
  can score 1.0 while being confidently wrong about GoDaddy's actual policy.
- **Faithfulness saturates.** All three configs sit near the ceiling, so that
  metric barely discriminates between them. Context precision is doing most of the
  real work in this comparison.
- **Dense retrieval only.** No BM25, no hybrid, no re-ranking — so this says
  nothing about whether those would help.
- **Three articles are titled "Unknown"** — the scraper didn't find a title. They
  still contribute chunks.

## What would actually improve this

In the order I'd do it:

1. Ground-truth answers and 30–50 questions, so the numbers can support a claim.
2. A judge from a different model family than the generator, to cut self-eval bias.
3. A BM25 + dense hybrid arm — the obvious next config, and the one most likely to
   beat naive on this corpus, since help-center questions carry exact product
   nouns that lexical search is good at.
4. A cross-encoder re-ranker over the fused candidates.

## License

MIT.
