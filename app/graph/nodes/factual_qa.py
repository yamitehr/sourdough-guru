"""Factual Q&A generation node."""

import logging

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.graph.state import SourdoughState
from app.graph.nodes.llm_utils import get_llm

logger = logging.getLogger("sourdough.factual_qa")

SYSTEM_PROMPT = """You are the Sourdough Guru, an expert sourdough baking assistant.

CRITICAL: You must ONLY use information from the provided context documents to answer. Do NOT use your general training knowledge about sourdough or baking. Every factual claim must come from the provided context.

Rules:
- ONLY state facts that are explicitly supported by the provided context
- ALWAYS cite the source for each claim (e.g., "According to *Tartine Bread*, p.52...")
- If the context does not contain enough information to fully answer, say: "Based on the available sources, I can tell you that [what context covers], but I don't have enough information in my knowledge base to address [what's missing]."
- Never invent temperatures, times, ratios, or techniques not found in the context
- Use proper baking terminology
- Include temperatures in both Celsius and Fahrenheit when the source provides them

Formatting:
- Use **Markdown** formatting for readability
- Use **bold** for key terms, temperatures, and important values
- Use ## headings to organize sections (e.g., ## Short Answer, ## Details)
- Use bullet points for lists
- Use > blockquotes for direct quotes from sources
- End with a **Sources** section listing the documents referenced"""


def _format_context(docs: list[dict]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.get("source", "Unknown")
        page = doc.get("page", "?")
        parts.append(f"[{i}] Source: {source}, Page {page}\n{doc['text']}")
    return "\n\n".join(parts)


def generate_qa_answer(state: SourdoughState) -> dict:
    """Generate a factual answer grounded in retrieved context."""
    llm = get_llm()

    context = _format_context(state.get("retrieved_docs", []))

    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    for msg in state.get("messages", [])[-6:]:
        role = getattr(msg, "type", "user")
        content = getattr(msg, "content", str(msg))
        if role == "human":
            messages.append(HumanMessage(content=content))
        elif role == "ai":
            messages.append(AIMessage(content=content))

    user_prompt = f"""Context from knowledge base:
{context}

User question: {state['user_query']}

Provide a thorough, grounded answer:"""

    messages.append(HumanMessage(content=user_prompt))

    logger.info(f"[FactualQA] Generating answer for: {state['user_query']}")

    response = llm.invoke(messages)
    answer = response.content.strip()

    logger.info(f"[FactualQA] Response length: {len(answer)} chars")
    logger.info(f"[FactualQA] Response preview: {answer[:200]}")

    step = {
        "module": "FactualQAAgent",
        "prompt": user_prompt,
        "response": answer,
    }

    return {
        "response": answer,
        "steps": [step],
    }
