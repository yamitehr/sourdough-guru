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
    try:
        result = llm.invoke(messages, max_tokens=200)
        translated = result.content.strip()
        if not translated:
            logger.warning("[Retriever] Translation returned empty result, using original query")
            return text
        logger.info(f"[Retriever] Translated query: '{text[:80]}' → '{translated[:80]}'")
        return translated
    except Exception as e:
        logger.warning(f"[Retriever] Translation failed: {e}, using original query")
        return text


MIN_RELEVANCE_SCORE = 0.30


def _build_search_query(state: SourdoughState) -> str:
    """Build a domain-relevant search query based on intent and params.

    For bake_plan/recipe intents, the raw user query is often a scheduling
    sentence (e.g., "plan 3 rye loaves by 6am") which embeds poorly against
    technique documents. Construct a targeted query instead.
    """
    intent = state.get("intent", "")
    params = state.get("intent_params", {})
    product = params.get("target_product", "")

    if intent in ("bake_plan", "recipe") and product:
        return f"sourdough {product} bread baking technique fermentation shaping"

    return state["user_query"]


def retrieve_context(state: SourdoughState) -> dict:
    """Embed the user query and retrieve top-k relevant chunks from Pinecone."""
    raw_query = _build_search_query(state)
    query = _translate_to_english(raw_query)

    logger.info(f"[Retriever] Querying Pinecone for: {query[:100]}")

    try:
        docs = retrieve(query=query, top_k=10)
    except Exception as e:
        logger.error(f"[Retriever] Pinecone retrieval failed: {e}")
        docs = []

    filtered_docs = []
    for i, doc in enumerate(docs):
        score = doc.get("score")
        score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "?"
        kept = isinstance(score, (int, float)) and score >= MIN_RELEVANCE_SCORE
        logger.info(
            f"[Retriever] Doc {i+1}: {doc.get('source', '?')} "
            f"(page {doc.get('page', '?')}, score={score_str}, {'KEPT' if kept else 'DROPPED'})"
        )
        if kept:
            filtered_docs.append(doc)

    translated_note = f" (translated from: {state['user_query'][:60]})" if query != raw_query else ""
    rewritten_note = f" (rewritten: {raw_query[:60]})" if raw_query != state["user_query"] else ""
    step = {
        "module": "retriever",
        "prompt": query,
        "response": f"Retrieved {len(filtered_docs)} documents (dropped {len(docs) - len(filtered_docs)} below score {MIN_RELEVANCE_SCORE}){rewritten_note}{translated_note}",
    }

    return {
        "retrieved_docs": filtered_docs,
        "steps": [step],
    }
