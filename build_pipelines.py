import json
import time
import os
from dotenv import load_dotenv
import anthropic
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter

import retrieval

load_dotenv()

# ---- SETUP ----
with open("articles.json") as f:
    articles = json.load(f)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")
db = chromadb.Client()

SYSTEM_PROMPT = """You are a GoDaddy customer support assistant.
Answer the question using only the provided context.
If the context doesn't contain enough information, say so.
Be concise and direct."""

# ---- CHUNKING ----
def chunk_articles(chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = []
    for article in articles:
        splits = splitter.split_text(article["content"])
        for i, split in enumerate(splits):
            chunks.append({
                "text": split,
                "source": article["url"],
                "title": article["title"],
                "id": f"{abs(hash(article['url']))}_{i}"
            })
    return chunks

# ---- BUILD VECTOR STORE ----
def build_collection(name, chunks):
    try:
        db.delete_collection(name)
    except Exception:
        pass
    collection = db.create_collection(name)
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()
    collection.add(
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"source": c["source"], "title": c["title"]} for c in chunks],
        ids=[c["id"] for c in chunks]
    )
    print(f"  '{name}': {len(chunks)} chunks from {len(articles)} articles")
    return collection

# ---- RETRIEVAL ----
def multi_query_variants(query, n=2):
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=150,
        messages=[{"role": "user", "content":
            f"Write {n} alternative phrasings of this question. "
            f"Return only the questions, one per line:\n{query}"}]
    )
    rewrites = [v for v in response.content[0].text.strip().split("\n") if v.strip()][:n]
    return [query] + rewrites


def multi_query_rerank(collection, cross_encoder, query, k, pool_per_variant=5):
    seen, candidates = set(), []
    for variant in multi_query_variants(query):
        for chunk in retrieval.dense_retrieve(collection, embedder, variant, pool_per_variant):
            if chunk not in seen:
                seen.add(chunk)
                candidates.append(chunk)
    return retrieval.rerank_retrieve(candidates, cross_encoder, query, k)

# ---- GENERATION ----
def generate(question, context_chunks):
    context = "\n\n---\n\n".join(context_chunks)
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=400,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}]
    )
    return response.content[0].text

# ---- RUN A CONFIG ----
def run_rag(config_name, retrieve_fn, question, k):
    start = time.time()
    chunks = retrieve_fn(question, k)
    answer = generate(question, chunks)
    return {
        "config": config_name,
        "question": question,
        "answer": answer,
        "contexts": chunks,
        "latency": round(time.time() - start, 2)
    }

# ---- BUILD ALL THREE CONFIGS ----
print("Building vector stores...")
# Config A — small non-overlapping chunks + hybrid retrieval (dense + BM25 / RRF)
chunks_a = chunk_articles(chunk_size=500,  chunk_overlap=0)
coll_a = build_collection("config_a_hybrid", chunks_a)
bm25_a = retrieval.BM25Index([c["text"] for c in chunks_a])

# Config B — larger overlapping chunks + semantic retrieval only
chunks_b = chunk_articles(chunk_size=1000, chunk_overlap=200)
coll_b = build_collection("config_b_semantic", chunks_b)

# Config C — small chunks + multi-query rewrite + cross-encoder re-ranker
chunks_c = chunk_articles(chunk_size=500,  chunk_overlap=0)
coll_c = build_collection("config_c_rerank", chunks_c)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ---- TEST WITH ONE QUESTION ----
test_q = "How do I turn off auto-renew for my domain?"

print(f"\nTest question: '{test_q}'\n")

r_a = run_rag("Config A - Hybrid (dense+BM25 RRF)",
              lambda q, k: retrieval.hybrid_retrieve(coll_a, embedder, bm25_a, q, k), test_q, k=6)
r_b = run_rag("Config B - Semantic only",
              lambda q, k: retrieval.dense_retrieve(coll_b, embedder, q, k), test_q, k=3)
r_c = run_rag("Config C - MultiQuery + Re-ranker",
              lambda q, k: multi_query_rerank(coll_c, cross_encoder, q, k), test_q, k=6)

for r in [r_a, r_b, r_c]:
    print(f"--- {r['config']} ({r['latency']}s) ---")
    print(r['answer'])
    print(f"Chunks retrieved: {len(r['contexts'])}\n")
