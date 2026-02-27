"""Simple script to sanity-check dependencies and APIs used by the project."""

import os

from dotenv import load_dotenv


def main() -> None:
    load_dotenv()

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("WARNING: ANTHROPIC_API_KEY is not set in environment")
    else:
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)
            message = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=50,
                messages=[{"role": "user", "content": "Say 'setup confirmed' and nothing else."}],
            )
            print("Claude API:", message.content[0].text)
        except Exception as exc:
            print("Claude API check failed:", exc)

    try:
        import chromadb

        db = chromadb.Client()
        _ = db.create_collection("test_setup")
        print("ChromaDB: OK")
    except Exception as exc:
        print("ChromaDB check failed:", exc)

    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb = model.encode("test sentence")
        print(f"Embeddings: OK — shape {emb.shape}")
    except Exception as exc:
        print("Embedding model check failed:", exc)


if __name__ == "__main__":
    main()
