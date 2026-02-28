"""Factual Q&A generation node."""

import logging

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.graph.state import SourdoughState, HISTORY_WINDOW
from app.graph.nodes.llm_utils import get_llm

logger = logging.getLogger("sourdough.factual_qa")

SYSTEM_PROMPT = """You are the Sourdough Guru, an expert sourdough baking assistant.

You are given context documents retrieved from a sourdough knowledge base. Use them as your primary source and always cite them. When the retrieved context does not fully cover the question, supplement with your own sourdough expertise — but clearly distinguish between what comes from the sources and what is general knowledge.

Rules:
- Prefer retrieved context and cite every claim drawn from it (e.g., "According to *Tartine Bread*, p.52...")
- When the context is silent on a point but the question IS about sourdough or baking, answer from your general sourdough knowledge and label it clearly (e.g., "Generally speaking..." or "As a rule of thumb...")
- If the retrieved context is empty AND the question is clearly NOT about sourdough or baking (e.g., politics, weather, sports, news), do NOT answer the question. Instead reply: "I'm the Sourdough Guru — that's outside my expertise! Can I help you with sourdough questions, a recipe, or planning your next bake?"
- Never refuse to answer a genuine sourdough question just because the exact topic isn't in the retrieved documents
- Never invent specific numbers (temperatures, times, ratios) that contradict the sources
- Use proper baking terminology
- Include temperatures in both Celsius and Fahrenheit when the source provides them
- Reply in the same language the user used

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
        parts.append(f"[{i}] Source: {source}, Page {page}\n{doc.get('text', '')}")
    return "\n\n".join(parts)


def generate_qa_answer(state: SourdoughState) -> dict:
    """Generate a factual answer grounded in retrieved context."""
    llm = get_llm()

    context = _format_context(state.get("retrieved_docs", []))

    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    for msg in state.get("messages", [])[-HISTORY_WINDOW:]:
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
        "module": "factual_qa",
        "prompt": user_prompt,
        "response": answer,
    }

    return {
        "response": answer,
        "steps": [step],
    }
