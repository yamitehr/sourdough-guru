"""Supervisor node: classify user intent and extract parameters via structured output."""

import json
import logging
import warnings
from datetime import datetime, timedelta
from typing import Optional, Literal

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.graph.state import SourdoughState
from app.graph.nodes.llm_utils import get_llm

logger = logging.getLogger("sourdough.supervisor")

SYSTEM_PROMPT_TEMPLATE = """You are the routing supervisor for a sourdough baking assistant.

Classify the user's message into exactly one intent and extract relevant parameters.

Intents:
- "factual_qa": Questions specifically about sourdough or bread baking — science, techniques, troubleshooting, ingredients, fermentation, equipment, history of sourdough.
- "recipe": Requests for a sourdough recipe, ingredient list, or baking formula.
- "bake_plan": Requests to plan/schedule a bake day with specific timing.
- "general": Everything else — greetings, meta-questions about the assistant, weather, news, politics, sports, and ANY question that is NOT about sourdough or bread baking.

Parameter rules:
- Extract numeric values as plain numbers (e.g. 75 not "75%").
- For time parameters: ALWAYS convert to ISO 8601 datetime format (YYYY-MM-DDTHH:MM:SS). Today is {today}. Tomorrow is {tomorrow}.
- Use "ready_by" when the user specifies when they want the bake FINISHED (e.g., "ready by 7am", "I need them at 6am", "done by morning").
- Use "start_time" when the user specifies when they want to START baking (e.g., "I'll start at 9am", "start now", "beginning at 8am").
- Never extract both from the same message — use whichever the user actually stated.
- The current date and time is {now} (local time). When the user says "start now", "right now", "immediately", or similar, use exactly {now} as the start_time.
- Example: "ready by 7am tomorrow" → ready_by: "{tomorrow}T07:00:00". "start at 9am tomorrow" → start_time: "{tomorrow}T09:00:00". "start now" → start_time: "{now}".
- When the user's message is a short follow-up or confirmation (e.g., "start now", "ok let's go", "yes", "do that") in a conversation where baking parameters were already established, carry over ALL previously stated parameters (num_loaves, temperature_c, hydration, flour_type, etc.) from the conversation history — do not drop them just because the current message doesn't repeat them."""


class IntentParams(BaseModel):
    target_product: Optional[str] = None
    hydration: Optional[float] = None
    flour_type: Optional[str] = None
    num_loaves: Optional[int] = None
    flour_g: Optional[float] = None
    starter_pct: Optional[float] = None
    salt_pct: Optional[float] = None
    temperature_c: Optional[float] = None
    ready_by: Optional[str] = None
    start_time: Optional[str] = None
    constraints: Optional[str] = None


class IntentClassification(BaseModel):
    """Classify user intent and extract baking parameters."""
    intent: Literal["factual_qa", "recipe", "bake_plan", "general"] = Field(
        description="The classified intent of the user's message"
    )
    intent_params: IntentParams = Field(
        default_factory=IntentParams,
        description="Extracted parameters relevant to the intent"
    )


def supervisor(state: SourdoughState) -> dict:
    """Classify intent and extract parameters using LangChain structured output."""
    llm = get_llm()
    structured_llm = llm.with_structured_output(IntentClassification)

    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    now_str = now.strftime("%Y-%m-%dT%H:%M:%S")
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(today=today, tomorrow=tomorrow, now=now_str)
    messages = [SystemMessage(content=system_prompt)]

    for msg in state.get("messages", [])[-6:]:
        role = getattr(msg, "type", "user")
        content = getattr(msg, "content", str(msg))
        if role == "human":
            messages.append(HumanMessage(content=content))
        elif role == "ai":
            messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=state["user_query"]))

    logger.info(f"[Supervisor] Query: {state['user_query']}")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result: IntentClassification = structured_llm.invoke(messages)
        intent = result.intent
        intent_params = result.intent_params.model_dump(exclude_none=True)
    except Exception as e:
        logger.warning(f"[Supervisor] Structured output failed: {e}, falling back to general")
        intent = "general"
        intent_params = {}

    logger.info(f"[Supervisor] Intent: {intent} | Params: {intent_params}")

    step = {
        "module": "supervisor",
        "prompt": state["user_query"],
        "response": json.dumps({"intent": intent, "intent_params": intent_params}),
    }

    return {
        "intent": intent,
        "intent_params": intent_params,
        "steps": [step],
    }
