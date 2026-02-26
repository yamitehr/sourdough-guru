"""RAG retrieval node: query Pinecone for relevant context."""

import logging

from app.graph.state import SourdoughState
from app.tools.knowledge_base import retrieve

logger = logging.getLogger("sourdough.retriever")


def retrieve_context(state: SourdoughState) -> dict:
    """Embed the user query and retrieve top-k relevant chunks from Pinecone."""
    query = state["user_query"]
    intent_params = state.get("intent_params", {})

    pinecone_filter = None
    if intent_params.get("source_type"):
        pinecone_filter = {"type": intent_params["source_type"]}

    logger.info(f"[Retriever] Querying Pinecone for: {query[:100]}")

    docs = retrieve(query=query, top_k=4, filter=pinecone_filter)

    for i, doc in enumerate(docs):
        logger.info(f"[Retriever] Doc {i+1}: {doc.get('source', '?')} (page {doc.get('page', '?')}, score={doc.get('score', '?'):.3f})")

    step = {
        "module": "KnowledgeBaseRetriever",
        "prompt": query,
        "response": f"Retrieved {len(docs)} documents",
    }

    return {
        "retrieved_docs": docs,
        "steps": [step],
    }
