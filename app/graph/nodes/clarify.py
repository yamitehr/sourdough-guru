"""Clarification node: check for missing parameters and ask the user before proceeding."""

import logging

from app.graph.state import SourdoughState
from app.tools.baking_math import BREAD_TYPES, normalize_product_type

logger = logging.getLogger("sourdough.clarify")

# Formatted bread-type options for clarification prompts.
# Country Loaf has a fully deterministic plan; other types require a recipe in the knowledge base.
_BREAD_TYPE_OPTIONS = (
    "  - Country Loaf (Pain de Campagne) ← full plan always available\n"
    "  - Any other sourdough product (focaccia, ciabatta, rolls, etc.) ← knowledge-base driven"
)

# Required parameters per intent — if any are missing, ask the user
REQUIRED_PARAMS = {
    "bake_plan": {
        "target_product": (
            f"What type of sourdough bread would you like to bake?\n{_BREAD_TYPE_OPTIONS}"
        ),
        "num_loaves": "How many loaves (or units) do you need?",
        "ready_by": "When do you need them ready, or when would you like to start? (e.g., 'ready by 7am tomorrow', 'start at 9am Saturday')",
        "temperature_c": "What's your kitchen temperature in °C? (This affects fermentation timing significantly)",
    },
    "recipe": {
        "target_product": "What type of sourdough product? (e.g., country loaf, focaccia, bagels, pizza dough)",
        "hydration": "What hydration level? (e.g., 70% for beginners, 80% for open crumb)",
    },
}


def clarify(state: SourdoughState) -> dict:
    """Check if critical parameters are missing and generate a clarifying question."""
    intent = state.get("intent", "general")
    params = state.get("intent_params", {})

    required = REQUIRED_PARAMS.get(intent)
    if not required:
        return {}

    missing = []
    for key, question in required.items():
        if key not in params or params[key] is None:
            # start_time is an acceptable alternative to ready_by for bake_plan
            if key == "ready_by" and intent == "bake_plan" and params.get("start_time"):
                continue
            # If target_product is provided but unrecognized, let it through —
            # the retriever + LLM will try to build a plan from knowledge base docs
            if key == "target_product" and intent == "bake_plan" and params.get("target_product"):
                continue
            missing.append(question)

    if not missing:
        logger.info(f"[Clarify] All required params present for {intent}")
        return {}

    logger.info(f"[Clarify] Missing params for {intent}: {[k for k in REQUIRED_PARAMS[intent] if k not in params]}")

    # Build a friendly clarifying response
    provided = {k: v for k, v in params.items() if v is not None}
    parts = []

    if intent == "bake_plan":
        parts.append("## Let's Plan Your Bake Day!\n")
        parts.append("I'd love to build you a detailed schedule. I just need a few details to get the timing right:\n")
    elif intent == "recipe":
        parts.append("## Let's Build Your Recipe!\n")
        parts.append("I'll create a detailed recipe with baker's percentages and step-by-step instructions. Just need a couple of details:\n")

    for q in missing:
        # Bold the first line (the question), leave any subsequent lines (e.g. options list) plain
        lines = q.split("\n", 1)
        if len(lines) > 1:
            parts.append(f"- **{lines[0]}**\n{lines[1]}")
        else:
            parts.append(f"- **{q}**")

    if provided:
        parts.append("\n**Got it so far:**")
        friendly = {
            "num_loaves": "Loaves",
            "ready_by": "Ready by",
            "temperature_c": "Kitchen temp",
            "target_product": "Product",
            "hydration": "Hydration",
            "flour_type": "Flour",
            "starter_pct": "Starter %",
            "start_time": "Start time",
            "salt_pct": "Salt %",
            "flour_g": "Flour weight",
        }
        for k, v in provided.items():
            label = friendly.get(k, k)
            # Show the resolved bread-type display name instead of the raw value
            if k == "target_product":
                resolved = normalize_product_type(str(v))
                if resolved:
                    v = BREAD_TYPES[resolved]["display_name"]
            parts.append(f"- {label}: **{v}**")

    response = "\n".join(parts)

    step = {
        "module": "clarify",
        "prompt": f"Intent: {intent}, Params: {params}",
        "response": response,
    }

    return {
        "response": response,
        "steps": [step],
    }
