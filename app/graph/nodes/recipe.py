"""Recipe recommendation generation node."""

import json
import logging

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.graph.state import SourdoughState, HISTORY_WINDOW
from app.graph.nodes.llm_utils import get_llm
from app.graph.nodes.param_utils import safe_float
from app.tools.baking_math import (
    BREAD_TYPES,
    normalize_product_type,
    calculate_bakers_percentages,
    calculate_hydration,
)

logger = logging.getLogger("sourdough.recipe")

SYSTEM_PROMPT = """You are the Sourdough Guru, an expert recipe creator for sourdough baking.

CRITICAL: Build the recipe primarily from the provided context documents and the baking math results. Prefer source-backed information and always cite it. When the sources cover a topic partially (e.g., they give a mixing method but not an oven temperature), fill in standard sourdough practice for the missing detail and note it briefly inline (e.g., "Bake at **230°C** (standard for this style)") — do NOT create a separate disclaimer section.

Your recipe MUST include:
1. Recipe name and brief description
2. Ingredients — use the baking math results provided (gram weights AND baker's percentages)
3. Method — step-by-step instructions drawn from the source techniques
4. Fermentation stages — times and sensory cues as described in the sources
5. Baking instructions — temperatures, steam, timing from the sources (supplement with standard practice if sources are silent)
6. Tips — practical advice found in the sources
7. Sources — list which books/papers each part of the recipe is based on

Rules:
- Use the baking math results for all ingredient quantities (do not recalculate)
- Cite the source for techniques, temperatures, and times (e.g., "as described in *Tartine Bread*")
- When the sources lack a specific detail, use standard sourdough technique and mark it briefly inline — never leave a section empty or create "what I cannot provide" blocks
- The recipe must always feel complete and actionable to the baker
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


def _extract_pcts_from_docs(
    docs: list[dict],
    product_name: str,
) -> dict[str, float | None]:
    """Ask the LLM to extract starter_pct and salt_pct from retrieved recipe documents.

    Returns {"starter_pct": float | None, "salt_pct": float | None}.
    Returns None values on any failure so the caller can fall back to defaults silently.
    """
    if not docs:
        return {"starter_pct": None, "salt_pct": None}

    llm = get_llm()
    context = "\n\n".join(
        f"[{d.get('source', '?')}]: {d.get('text', '')}" for d in docs
    )

    prompt = f"""You are a baking expert reading sourdough recipe source documents.

Product: {product_name}

Source documents:
{context}

---
Extract ONLY the following from the documents, as baker's percentages (relative to total flour weight):
- starter_pct: the sourdough starter / levain percentage (e.g. 20 means 20% of flour weight)
- salt_pct: the salt percentage (e.g. 2 means 2% of flour weight)

Rules:
- If a percentage is explicitly stated in the text (e.g. "20% starter", "2% salt"), use it directly.
- If only gram weights are given for both salt/starter AND flour, compute the percentage yourself.
- If you cannot determine a value with confidence, set it to null.
- Do NOT guess or hallucinate. Return null if unsure.

Return ONLY valid JSON, no markdown fences, no extra text:
{{"starter_pct": <number or null>, "salt_pct": <number or null>}}"""

    try:
        response = llm.invoke(
            [
                SystemMessage(
                    content="You are a baking expert. Return only valid JSON."
                ),
                HumanMessage(content=prompt),
            ]
        )
        raw = response.content.strip()

        # Strip accidental markdown fences
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        data = json.loads(raw)
        if not isinstance(data, dict):
            return {"starter_pct": None, "salt_pct": None}

        def _to_float_or_none(val) -> float | None:
            if val is None:
                return None
            try:
                f = float(val)
                return f if f > 0 else None
            except (TypeError, ValueError):
                return None

        result = {
            "starter_pct": _to_float_or_none(data.get("starter_pct")),
            "salt_pct": _to_float_or_none(data.get("salt_pct")),
        }
        logger.info(
            f"[RecipePcts] RAG-extracted for '{product_name}': "
            f"starter_pct={result['starter_pct']}, salt_pct={result['salt_pct']}"
        )
        return result

    except Exception as e:
        logger.info(f"[RecipePcts] Could not extract pcts from docs: {e}")
        return {"starter_pct": None, "salt_pct": None}


def compute_baking_math(state: SourdoughState) -> dict:
    """Run deterministic baking math calculations."""
    params = state.get("intent_params", {})

    # Resolve bread type config for defaults (mirrors bake_plan.py approach)
    raw_product = params.get("target_product", "")
    product_type = normalize_product_type(raw_product or "") or "country_loaf"
    bread_config = BREAD_TYPES.get(product_type, BREAD_TYPES["country_loaf"])

    flour_g = safe_float(params.get("flour_g"), 1000)
    hydration = safe_float(params.get("hydration"), bread_config["default_hydration"])

    # For starter_pct and salt_pct: priority is
    #   1. User-supplied value (already in intent_params)
    #   2. RAG-extracted value from retrieved documents
    #   3. Bread-type config default
    retrieved_docs = state.get("retrieved_docs", [])
    rag_pcts: dict[str, float | None] = {"starter_pct": None, "salt_pct": None}
    rag_pct_step = None
    needs_rag = "starter_pct" not in params or "salt_pct" not in params
    if needs_rag and retrieved_docs:
        rag_pcts = _extract_pcts_from_docs(retrieved_docs, raw_product or product_type)
        rag_pct_step = {
            "module": "recipe_pct_extraction",
            "prompt": f"Extract starter_pct and salt_pct for '{raw_product or product_type}' from {len(retrieved_docs)} docs",
            "response": json.dumps(rag_pcts),
        }

    if "starter_pct" in params:
        starter_pct = safe_float(
            params["starter_pct"], bread_config["default_starter_pct"]
        )
    elif rag_pcts["starter_pct"] is not None:
        starter_pct = rag_pcts["starter_pct"]
    else:
        starter_pct = bread_config["default_starter_pct"]

    if "salt_pct" in params:
        salt_pct = safe_float(params["salt_pct"], bread_config["default_salt_pct"])
    elif rag_pcts["salt_pct"] is not None:
        salt_pct = rag_pcts["salt_pct"]
    else:
        salt_pct = bread_config["default_salt_pct"]

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

    result: dict = {"math_results": math_results}
    if rag_pct_step:
        result["steps"] = [rag_pct_step]
    return result


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

User request: {state["user_query"]}

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
