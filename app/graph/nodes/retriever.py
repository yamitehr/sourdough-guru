"""RAG retrieval node: query Pinecone for relevant context."""

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.graph.state import SourdoughState
from app.graph.nodes.llm_utils import get_llm
from app.tools.knowledge_base import retrieve

logger = logging.getLogger("sourdough.retriever")


def _is_likely_english(text: str) -> bool:
    """Heuristic: if >80% of chars are ASCII, treat as English."""
    if not text:
        return True
    return sum(1 for c in text if ord(c) < 128) / len(text) > 0.8


def _translate_to_english(text: str) -> str:
    """Translate query to English for better embedding alignment with the knowledge base."""
    if _is_likely_english(text):
        return text

    llm = get_llm()
    messages = [
        SystemMessage(content="Translate the following text to English. Return only the translated text, nothing else."),
        HumanMessage(content=text),
    ]
    result = llm.invoke(messages, max_tokens=200)
    translated = result.content.strip()
    logger.info(f"[Retriever] Translated query: '{text[:80]}' → '{translated[:80]}'")
    return translated


def retrieve_context(state: SourdoughState) -> dict:
    """Embed the user query and retrieve top-k relevant chunks from Pinecone."""
    query = _translate_to_english(state["user_query"])
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
