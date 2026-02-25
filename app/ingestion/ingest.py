"""CLI entry point: parse PDFs → chunk → embed → upsert to Pinecone."""

import os
import sys
import hashlib
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

from app.ingestion.pdf_parser import extract_pages
from app.ingestion.chunker import chunk_pages

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

BATCH_SIZE = 100  # vectors per upsert batch
EMBED_BATCH_SIZE = 50  # texts per embedding call


def get_embedding_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["LLMOD_API_KEY"],
        base_url=os.environ.get("LLMOD_BASE_URL", "https://api.llmod.ai/v1"),
    )


def embed_texts(client: OpenAI, texts: list[str], model: str) -> list[list[float]]:
    """Embed a batch of texts."""
    resp = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in resp.data]


def make_id(source: str, chunk_index: int) -> str:
    """Deterministic vector ID from source filename + chunk index."""
    raw = f"{source}::{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


def ingest_directory(directory: Path, doc_type: str, index, client: OpenAI, model: str):
    """Process all PDFs in a directory."""
    pdf_files = sorted(directory.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs in {directory} (type={doc_type})")

    for pdf_path in pdf_files:
        print(f"\n--- Processing: {pdf_path.name}")
        try:
            pages = extract_pages(pdf_path)
        except Exception as e:
            print(f"  ERROR parsing {pdf_path.name}: {e}")
            continue

        print(f"  Extracted {len(pages)} pages")
        chunks = chunk_pages(pages, doc_type=doc_type)
        print(f"  Split into {len(chunks)} chunks")

        if not chunks:
            continue

        # Embed and upsert in batches
        for i in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[i : i + EMBED_BATCH_SIZE]
            texts = [c["text"] for c in batch]

            try:
                embeddings = embed_texts(client, texts, model)
            except Exception as e:
                print(f"  ERROR embedding batch {i}: {e}")
                continue

            vectors = []
            for chunk, embedding in zip(batch, embeddings):
                vec_id = make_id(chunk["metadata"]["source"], chunk["metadata"]["chunk_index"])
                vectors.append({
                    "id": vec_id,
                    "values": embedding,
                    "metadata": {**chunk["metadata"], "text": chunk["text"]},
                })

            # Upsert to Pinecone
            for j in range(0, len(vectors), BATCH_SIZE):
                upsert_batch = vectors[j : j + BATCH_SIZE]
                index.upsert(vectors=upsert_batch)

            print(f"  Upserted {len(vectors)} vectors (batch starting at chunk {i})")

    print(f"\nDone with {directory}")


def main():
    dataset_root = Path(__file__).resolve().parent.parent.parent / "dataset"
    books_dir = dataset_root / "Books"
    papers_dir = dataset_root / "Research papers"

    if not dataset_root.exists():
        print(f"Dataset directory not found: {dataset_root}")
        sys.exit(1)

    # Initialize clients
    client = get_embedding_client()
    model = os.environ.get("LLMOD_EMBEDDING_MODEL", "RPRTHPB-text-embedding-3-small")

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = os.environ.get("PINECONE_INDEX", "sourdough-knowledge")

    # Create index if it doesn't exist
    existing = [i.name for i in pc.list_indexes()]
    if index_name not in existing:
        print(f"Creating Pinecone index '{index_name}' (dimension=1536, cosine)...")
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print("Index created. Waiting for it to be ready...")
        while not pc.describe_index(index_name).status["ready"]:
            import time
            time.sleep(1)
        print("Index ready.")

    index = pc.Index(index_name)

    print("=== Sourdough Guru Ingestion Pipeline ===\n")

    if books_dir.exists():
        ingest_directory(books_dir, "book", index, client, model)

    if papers_dir.exists():
        ingest_directory(papers_dir, "research_paper", index, client, model)

    print("\n=== Ingestion complete ===")


if __name__ == "__main__":
    main()
