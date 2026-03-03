"""Supervisor node: classify user intent and extract parameters via structured output."""

import json
import logging
import warnings
from datetime import datetime, timedelta
from typing import Optional, Literal

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.graph.state import SourdoughState, HISTORY_WINDOW
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
- temperature_c must ALWAYS be in Celsius. If the user provides Fahrenheit (e.g., "72°F", "72 degrees"), convert to Celsius: (F-32)×5/9. Example: "72°F" → temperature_c: 22.
- Extract starter_pct whenever the user mentions a starter, levain, or preferment percentage (e.g. "15% starter", "20% levain", "use 25% preferment" → starter_pct: 25).
- Extract salt_pct whenever the user mentions a salt percentage (e.g. "2.5% salt", "3 percent salt", "salt at 1.8%" → salt_pct: 2.5 / 3 / 1.8).
- For time parameters: ALWAYS convert to ISO 8601 datetime format (YYYY-MM-DDTHH:MM:SS). Today is {today}. Tomorrow is {tomorrow}.
- Use "ready_by" when the user specifies when they want the bake FINISHED (e.g., "ready by 7am", "I need them at 6am", "done by morning").
- Use "start_time" when the user specifies when they want to START baking (e.g., "I'll start at 9am", "start now", "beginning at 8am").
- If the user explicitly states BOTH a start time AND a finish/ready-by time (e.g. "start at 6pm, done by 8pm"), extract BOTH start_time and ready_by. Otherwise extract only the one the user stated.
- The current date and time is {now} (local time). When the user says "start now", "right now", "immediately", or similar, use exactly {now} as the start_time.
- For relative finish times ("end in 2 hours", "done in 3 hours", "finish in 90 minutes", "ready in 2 hours"): convert to an absolute ready_by by adding that duration to the current time {now}.
- Example: "ready by 7am tomorrow" → ready_by: "{tomorrow}T07:00:00". "start at 9am tomorrow" → start_time: "{tomorrow}T09:00:00". "start now" → start_time: "{now}". "start now and done in 2 hours" → start_time: "{now}", ready_by: "{in_2h}". "done in 3 hours" → ready_by: "{in_3h}".
- When the user's message is a short follow-up or confirmation (e.g., "start now", "ok let's go", "yes", "do that") in a conversation where baking parameters were already established, carry over ALL previously stated parameters (num_loaves, temperature_c, hydration, flour_type, etc.) from the conversation history — do not drop them just because the current message doesn't repeat them.
- CRITICAL — answering a question: When the assistant's last message asked the user a specific question (e.g., "What's your kitchen temperature?"), interpret the user's reply as the answer to THAT question. For example, if the assistant asked for kitchen temperature and the user replies "22", that means temperature_c=22, NOT a time or any other parameter. Always check the last assistant message for context before interpreting short replies.
- CRITICAL — switching products: When the user says they want to bake "something else", "a different bread", "switch", or otherwise indicates they want to CHANGE the product type, do NOT carry over target_product from the conversation history. Leave target_product empty so the assistant can ask what they want instead. You may still carry over other parameters like num_loaves and temperature_c if they haven't been explicitly changed.
- CRITICAL — always extract target_product when the user names any bread or baked good, even generic terms like "loaf", "loaves", "bread", "rolls". These are explicit product specifications and must be extracted — they will override the previously selected product via the merging logic. Example: if the previous product was "cinnamon rolls" and the user now says "4 loafs", extract target_product: "loaf"."""


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
        description="Extracted parameters relevant to the intent",
    )


def supervisor(state: SourdoughState) -> dict:
    """Classify intent and extract parameters using LangChain structured output."""
    llm = get_llm()
    structured_llm = llm.with_structured_output(IntentClassification)

    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    now_str = now.strftime("%Y-%m-%dT%H:%M:%S")
    in_1h = (now + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S")
    in_2h = (now + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S")
    in_3h = (now + timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%S")
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        today=today,
        tomorrow=tomorrow,
        now=now_str,
        in_1h=in_1h,
        in_2h=in_2h,
        in_3h=in_3h,
    )
    messages = [SystemMessage(content=system_prompt)]

    for msg in state.get("messages", [])[-HISTORY_WINDOW:]:
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
        intent_params = {
            k: v
            for k, v in result.intent_params.model_dump(exclude_none=True).items()
            if v != ""
        }
    except Exception as e:
        logger.warning(
            f"[Supervisor] Structured output failed: {e}, falling back to general"
        )
        intent = "general"
        intent_params = {}

    # Deterministic param merging: if the intent is the same as the previous turn,
    # merge new params ON TOP of old params so values aren't lost across turns.
    # New non-empty values override; old values are preserved if not re-extracted.
    prev_intent = state.get("intent", "")
    prev_params = state.get("intent_params", {})
    if intent == prev_intent and prev_params:
        # When the product is changing, carry over universal params:
        # num_loaves, temperature_c, timing, and user-specified salt/starter percentages.
        # Product-specific params like hydration and flour_g reset to the new type's defaults.
        prev_product = prev_params.get("target_product", "")
        new_product = intent_params.get("target_product", "")
        product_changing = (
            new_product and prev_product and new_product.lower() != prev_product.lower()
        ) or (not new_product and prev_product)  # user wants "something else"

        if product_changing:
            _UNIVERSAL_PARAMS = {
                "num_loaves",
                "temperature_c",
                "start_time",
                "ready_by",
                "salt_pct",
                "starter_pct",
            }
            carry_over = {
                k: v for k, v in prev_params.items() if k in _UNIVERSAL_PARAMS
            }
            merged = {**carry_over, **intent_params}
            logger.info(
                f"[Supervisor] Product change — carrying over only universal params: {carry_over}"
            )
        else:
            merged = {**prev_params, **intent_params}
            logger.info(
                f"[Supervisor] Merged with prev params: {prev_params} + {intent_params} = {merged}"
            )
        intent_params = merged

    logger.info(f"[Supervisor] Intent: {intent} | Params: {intent_params}")

    step = {
        "module": "supervisor",
        "prompt": state["user_query"],
        "response": json.dumps({"intent": intent, "intent_params": intent_params}),
    }

    return {
        "intent": intent,
        "intent_params": intent_params,
        "response": "",  # clear stale response from previous turn
        "steps": [step],
    }
