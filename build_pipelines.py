import json
import time
import os
from dotenv import load_dotenv
import anthropic
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
                "id": f"{article['url']}_{i}"
            })
    return chunks

# ---- BUILD VECTOR STORE ----
def build_collection(name, chunk_size, chunk_overlap):
    try:
        db.delete_collection(name)
    except:
        pass
    collection = db.create_collection(name)
    chunks = chunk_articles(chunk_size, chunk_overlap)
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
def retrieve(collection, query, k):
    embedding = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=k)
    return results["documents"][0]

def multi_query_retrieve(collection, query, k):
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=150,
        messages=[{"role": "user", "content": 
            f"Write 2 alternative phrasings of this question. Return only the questions, one per line:\n{query}"}]
    )
    variants = [query] + response.content[0].text.strip().split("\n")[:2]
    seen, all_chunks = set(), []
    for q in variants:
        for chunk in retrieve(collection, q, k):
            if chunk not in seen:
                seen.add(chunk)
                all_chunks.append(chunk)
    return all_chunks[:k]

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
def run_rag(config_name, collection, question, k, multi_query=False):
    start = time.time()
    chunks = multi_query_retrieve(collection, question, k) if multi_query else retrieve(collection, question, k)
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
config1 = build_collection("naive_rag",     chunk_size=500,  chunk_overlap=0)
config2 = build_collection("optimized_rag", chunk_size=1000, chunk_overlap=200)
config3 = build_collection("multiquery_rag",chunk_size=1000, chunk_overlap=200)

# ---- TEST WITH ONE QUESTION ----
test_q = "How do I turn off auto-renew for my domain?"

print(f"\nTest question: '{test_q}'\n")

r1 = run_rag("Config 1 - Naive",      config1, test_q, k=3)
r2 = run_rag("Config 2 - Optimized",  config2, test_q, k=5)
r3 = run_rag("Config 3 - MultiQuery", config3, test_q, k=5, multi_query=True)

for r in [r1, r2, r3]:
    print(f"--- {r['config']} ({r['latency']}s) ---")
    print(r['answer'])
    print(f"Chunks retrieved: {len(r['contexts'])}\n")