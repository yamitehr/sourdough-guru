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
- "factual_qa": Questions about sourdough science, techniques, troubleshooting, ingredients, history.
- "recipe": Requests for a recipe, ingredient list, or formula.
- "bake_plan": Requests to plan/schedule a bake day with specific timing.
- "general": Greetings, meta-questions about the assistant, off-topic.

Parameter rules:
- Extract numeric values as plain numbers (e.g. 75 not "75%").
- For ready_by and start_time: ALWAYS convert to ISO 8601 datetime format (YYYY-MM-DDTHH:MM:SS). If the user says "6am" or "6am tomorrow", infer the next occurrence of that time relative to today ({today}) and output a full ISO datetime. Today is {today}. Tomorrow is {tomorrow}. Example: "6am tomorrow" → "{tomorrow}T06:00:00"."""


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
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(today=today, tomorrow=tomorrow)
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
