"""Pinecone retriever for sourdough knowledge base."""

from functools import lru_cache

from openai import OpenAI
from pinecone import Pinecone

from app.config import get_settings


@lru_cache
def _get_clients():
    settings = get_settings()
    embed_client = OpenAI(
        api_key=settings.LLMOD_API_KEY,
        base_url=settings.LLMOD_BASE_URL,
    )
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index = pc.Index(settings.PINECONE_INDEX)
    return embed_client, index


def retrieve(
    query: str,
    top_k: int = 5,
    filter: dict | None = None,
) -> list[dict]:
    """Retrieve relevant chunks from Pinecone.

    Returns a list of dicts with keys: text, source, page, type, score.
    """
    settings = get_settings()
    embed_client, index = _get_clients()

    resp = embed_client.embeddings.create(
        input=[query],
        model=settings.LLMOD_EMBEDDING_MODEL,
    )
    query_embedding = resp.data[0].embedding

    query_params = {
        "vector": query_embedding,
        "top_k": top_k,
        "include_metadata": True,
    }
    if filter:
        query_params["filter"] = filter

    results = index.query(**query_params)

    docs = []
    for match in results.matches:
        meta = match.metadata or {}
        docs.append({
            "text": meta.get("text", ""),
            "source": meta.get("source", ""),
            "page": meta.get("page", 0),
            "type": meta.get("type", ""),
            "score": match.score,
        })
    return docs
