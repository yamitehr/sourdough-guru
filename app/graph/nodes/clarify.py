"""Clarification node: check for missing parameters and ask the user before proceeding."""

import logging

from app.graph.state import SourdoughState

logger = logging.getLogger("sourdough.clarify")

# Required parameters per intent — if any are missing, ask the user
REQUIRED_PARAMS = {
    "bake_plan": {
        "num_loaves": "How many loaves do you need?",
        "ready_by": "When do you need them ready, or when would you like to start? (e.g., 'ready by 7am tomorrow', 'start at 9am Saturday')",
        "temperature_c": "What's your kitchen temperature? (This affects fermentation timing significantly)",
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
        }
        for k, v in provided.items():
            label = friendly.get(k, k)
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
