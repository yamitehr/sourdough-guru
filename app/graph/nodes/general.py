"""General/chitchat response node."""

import logging

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.graph.state import SourdoughState
from app.graph.nodes.llm_utils import get_llm

logger = logging.getLogger("sourdough.general")

SYSTEM_PROMPT = """You are the Sourdough Guru, a friendly sourdough baking assistant.

IMPORTANT: Always stay in character as the Sourdough Guru. Never discuss your internal workings, sessions, memory, or AI architecture. If the user asks meta-questions about sessions or how you work, briefly acknowledge and redirect to sourdough baking.

You can help with:
1. Factual Q&A — Answer questions about sourdough science, techniques, troubleshooting, and ingredients
2. Recipe Recommendations — Create customized sourdough recipes with baker's percentages
3. Bake Planning — Build detailed bake-day schedules with timestamps and notifications

For general greetings or off-topic questions, be friendly and steer the conversation toward sourdough baking.
Keep responses concise and helpful.

Formatting:
- Use **Markdown** formatting for readability
- Use **bold** for emphasis on key points
- Use bullet points for lists"""


def generate_general_response(state: SourdoughState) -> dict:
    """Handle greetings, meta-questions, and off-topic queries."""
    llm = get_llm()

    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    for msg in state.get("messages", [])[-6:]:
        role = getattr(msg, "type", "user")
        content = getattr(msg, "content", str(msg))
        if role == "human":
            messages.append(HumanMessage(content=content))
        elif role == "ai":
            messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=state["user_query"]))

    logger.info(f"[General] Generating response for: {state['user_query']}")

    response = llm.invoke(messages)
    answer = response.content.strip()

    logger.info(f"[General] Response length: {len(answer)} chars")
    logger.info(f"[General] Response preview: {answer[:200]}")

    if not answer:
        logger.warning(f"[General] Empty answer!")

    step = {
        "module": "GeneralAgent",
        "prompt": state["user_query"],
        "response": answer,
    }

    return {
        "response": answer,
        "steps": [step],
    }
