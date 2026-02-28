"""Recipe recommendation generation node."""

import json
import logging

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.graph.state import SourdoughState, HISTORY_WINDOW
from app.graph.nodes.llm_utils import get_llm
from app.graph.nodes.param_utils import safe_float
from app.tools.baking_math import (
    calculate_bakers_percentages,
    calculate_hydration,
)

logger = logging.getLogger("sourdough.recipe")

SYSTEM_PROMPT = """You are the Sourdough Guru, an expert recipe creator for sourdough baking.

CRITICAL: Build the recipe ONLY from the provided context documents and the baking math results. Do NOT use your general training knowledge. Every technique, temperature, time, tip, and ingredient amount must come from the provided sources.

Your recipe MUST include:
1. Recipe name and brief description
2. Ingredients — use the baking math results provided (gram weights AND baker's percentages)
3. Method — step-by-step instructions drawn from the source techniques
4. Fermentation stages — times and sensory cues as described in the sources
5. Baking instructions — temperatures, steam, timing from the sources
6. Tips — practical advice found in the sources
7. Sources — list which books/papers each part of the recipe is based on

Rules:
- Use the baking math results for all ingredient quantities (do not recalculate)
- Cite the source for techniques, temperatures, and times (e.g., "as described in *Tartine Bread*")
- If the retrieved context doesn't have enough information for a specific aspect (e.g., missing ingredient amounts, missing technique), say so explicitly — do NOT fill in from general knowledge. Instead, tell the user what you do have and suggest a recipe that IS well-covered in your sources
- The ingredient table must be clean: only Ingredient, Weight, and Baker's % columns — no source annotations inside the table

Formatting:
- Use **Markdown** formatting for readability
- Use ## headings for each recipe section (## Ingredients, ## Method, ## Fermentation, etc.)
- Use **bold** for key values (temperatures, times, weights)
- Use numbered lists for method steps
- Use a table for ingredients (| Ingredient | Weight | Baker's % |)
- End with a **Sources** section listing the documents used"""


def _format_context(docs: list[dict]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.get("source", "Unknown")
        parts.append(f"[{i}] {source}: {doc.get('text', '')}")
    return "\n\n".join(parts)


def compute_baking_math(state: SourdoughState) -> dict:
    """Run deterministic baking math calculations."""
    params = state.get("intent_params", {})

    flour_g = safe_float(params.get("flour_g"), 1000)
    hydration = safe_float(params.get("hydration"), 75)
    starter_pct = safe_float(params.get("starter_pct"), 20)
    salt_pct = safe_float(params.get("salt_pct"), 2)

    water_g = flour_g * (hydration / 100)
    starter_g = flour_g * (starter_pct / 100)
    salt_g = flour_g * (salt_pct / 100)

    ingredients = {
        "flour": flour_g,
        "water": water_g,
        "starter": starter_g,
        "salt": salt_g,
    }

    bakers_pct = calculate_bakers_percentages(
        {"water": water_g, "starter": starter_g, "salt": salt_g},
        flour_g,
    )

    actual_hydration = calculate_hydration(water_g, flour_g)

    math_results = {
        "ingredients": ingredients,
        "bakers_percentages": bakers_pct,
        "hydration": actual_hydration,
        "flour_g": flour_g,
    }

    logger.info(f"[BakingMath] hydration={actual_hydration}% flour={flour_g}g")

    return {"math_results": math_results}


def generate_recipe(state: SourdoughState) -> dict:
    """Generate a recipe using LLM with baking math and retrieved context."""
    llm = get_llm()

    context = _format_context(state.get("retrieved_docs", []))
    math = state.get("math_results", {})

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

Baking math results:
{json.dumps(math, indent=2)}

User request: {state['user_query']}

Create a detailed sourdough recipe:"""

    messages.append(HumanMessage(content=user_prompt))

    logger.info(f"[Recipe] Generating recipe for: {state['user_query']}")

    response = llm.invoke(messages)
    answer = response.content.strip()

    logger.info(f"[Recipe] Response length: {len(answer)} chars")
    logger.info(f"[Recipe] Response preview: {answer[:200]}")

    step = {
        "module": "recipe",
        "prompt": user_prompt,
        "response": answer,
    }

    return {
        "response": answer,
        "steps": [step],
    }
