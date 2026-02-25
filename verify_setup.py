import anthropic
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
message = client.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=50,
    messages=[{"role": "user", "content": "Say 'setup confirmed' and nothing else."}]
)
print("Claude API:", message.content[0].text)

db = chromadb.Client()
collection = db.create_collection("test")
print("ChromaDB: OK")

model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = model.encode("test sentence")
print(f"Embeddings: OK — shape {embedding.shape}")