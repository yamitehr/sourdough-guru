"""Bake planning node: build timeline and generate natural language plan."""

import json
import logging
import re
from datetime import datetime, timedelta

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.graph.state import SourdoughState, HISTORY_WINDOW
from app.graph.nodes.llm_utils import get_llm
from app.graph.nodes.param_utils import safe_float, safe_int
from app.tools.baking_math import (
    BREAD_TYPES,
    estimate_fermentation_time,
    normalize_product_type,
    compute_recipe,
    default_bread_steps,
    calculate_timeline,
    _parse_time,
)

logger = logging.getLogger("sourdough.bake_plan")

# timeline_source values stored in bake_plan_data:
#   "hardcoded"        — country loaf, deterministic template, no LLM for steps
#   "extracted"        — LLM successfully extracted steps + ingredients from KB recipe
#   "no_recipe_found"  — docs retrieved but no matching recipe inside them
#   "no_docs"          — nothing retrieved from KB at all

SYSTEM_PROMPT = """You are the Sourdough Guru, a bake-day planning expert.

CRITICAL: Use ONLY the provided baking timeline data and context documents. Do NOT add steps, temperatures, or techniques from your general training knowledge. The timeline was built from the recipe in the knowledge base — present it faithfully.

Given the baking timeline with concrete timestamps, create a clear, friendly bake plan:
1. Summarize the overall schedule (start time -> finish time)
2. Walk through each step with its timestamp and what to do
3. For sensory cues and tips, ONLY use information from the provided context documents and cite the source
4. Highlight critical timing points (when the baker MUST take action)
5. Note any steps where the baker can rest/sleep

Formatting:
- Use **Markdown** formatting for readability
- Use ## headings for major sections (## Overview, ## Ingredients, ## Timeline, ## Tips)
- Use **bold** for all timestamps and critical action items
- Use a table for the timeline overview (| Time | Step | Duration |)
- Mark rest/sleep periods with a relaxed tone (e.g., "You can sleep now!")
- Use > blockquotes for tips from sources
- End with a **Sources** section listing the documents referenced
- NEVER include developer/system notes about where data came from. Write as if YOU are the expert giving the plan directly to the baker."""


# ---------------------------------------------------------------------------
# Recipe extraction from knowledge base docs
# ---------------------------------------------------------------------------


def _extract_recipe_from_docs(
    docs: list[dict],
    product_name: str,
    bulk_hours: float,
    num_units: int,
) -> dict | None:
    """Call LLM to extract a complete recipe (ingredients + steps) from retrieved docs.

    Returns a dict:
        {
            "recipe_found": bool,
            "ingredients": [{"name": str, "amount": str}, ...],
            "steps": [{"name": str, "duration_minutes": int, "description": str}, ...]
        }
    Returns None on LLM / JSON-parse failure (caller treats as graceful fallback).
    """
    llm = get_llm()

    context = "\n\n".join(
        f"[{d.get('source', '?')}]: {d.get('text', '')}" for d in docs
    )
    bulk_minutes = int(bulk_hours * 60)

    prompt = f"""You are a baking expert extracting recipe information from source documents.

Product requested: **{product_name}**
Number of units to make: {num_units}
Bulk fermentation / first-rise time (already calculated from kitchen temperature): {bulk_hours} hours ({bulk_minutes} minutes)

Source documents:
{context}

---
YOUR TASK:

1. Search the documents for any recipe related to **{product_name}**. Be flexible with naming —
   e.g. "rye bread" matches "sourdough rye", "dark rye loaf", "rye sourdough", etc.
   Accept partial matches and related variations.

2. If ANY relevant recipe content is found (even partial — some ingredients or some steps),
   set "recipe_found" to true and extract whatever is available:
   a. Ingredients: extract what is mentioned, scaled to {num_units} unit(s). If an exact
      amount is not stated, make a reasonable estimate based on baker's percentages and note it.
   b. Steps: every step mentioned in the correct order, from mixing through cooling.
      Fill gaps with standard sourdough technique where the documents are silent on a step.

3. For any bulk fermentation / first rise step, use exactly {bulk_minutes} minutes.
4. Include temperatures, visual cues, and pan types where mentioned in the documents.
5. Only set "recipe_found" to false if the documents contain NO relevant content at all
   for this type of bread.

Return ONLY valid JSON in this exact structure (no markdown fences, no extra text):

{{
  "recipe_found": true,
  "ingredients": [
    {{"name": "dark rye flour", "amount": "300g"}},
    {{"name": "bread flour", "amount": "200g"}},
    {{"name": "water", "amount": "400g"}},
    {{"name": "sourdough starter", "amount": "100g"}},
    {{"name": "salt", "amount": "10g"}}
  ],
  "steps": [
    {{"name": "Mix dough", "duration_minutes": 15, "description": "Combine rye and bread flour with water; mix until no dry flour remains."}},
    {{"name": "Bulk fermentation", "duration_minutes": {bulk_minutes}, "description": "Cover and ferment. Rye dough is sticky — use wet hands for any folds."}}
  ]
}}

If truly no relevant content found, return:
{{"recipe_found": false, "ingredients": [], "steps": []}}"""

    try:
        response = llm.invoke(
            [
                SystemMessage(
                    content="You are a baking expert. Follow the instructions exactly and return only valid JSON."
                ),
                HumanMessage(content=prompt),
            ]
        )
        raw = response.content.strip()

        # Strip accidental markdown code fences
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        data = json.loads(raw)

        if not isinstance(data, dict):
            logger.warning("[RecipeExtraction] LLM returned non-dict JSON")
            return None

        recipe_found = bool(data.get("recipe_found", False))

        if not recipe_found:
            logger.info(
                f"[RecipeExtraction] LLM reports no recipe for '{product_name}' in docs"
            )
            return {"recipe_found": False, "ingredients": [], "steps": []}

        # Validate and sanitise steps
        valid_steps = []
        for s in data.get("steps", []):
            if not isinstance(s, dict):
                continue
            name = str(s.get("name", "")).strip()
            desc = str(s.get("description", "")).strip()
            try:
                dur = int(s["duration_minutes"])
            except (KeyError, ValueError, TypeError):
                continue
            if name and dur > 0:
                valid_steps.append(
                    {"name": name, "duration_minutes": dur, "description": desc}
                )

        # Validate and sanitise ingredients
        valid_ingredients = []
        for ing in data.get("ingredients", []):
            if not isinstance(ing, dict):
                continue
            name = str(ing.get("name", "")).strip()
            amount = str(ing.get("amount", "")).strip()
            if name:
                valid_ingredients.append({"name": name, "amount": amount})

        logger.info(
            f"[RecipeExtraction] '{product_name}': {len(valid_steps)} steps, "
            f"{len(valid_ingredients)} ingredients extracted"
        )

        return {
            "recipe_found": True,
            "ingredients": valid_ingredients,
            "steps": valid_steps,
        }

    except json.JSONDecodeError as e:
        logger.warning(
            f"[RecipeExtraction] JSON parse error: {e}. Raw: {response.content[:200] if 'response' in dir() else 'N/A'}"
        )
        return None
    except Exception as e:
        logger.warning(f"[RecipeExtraction] Extraction failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Baker's percentage parsing from extracted ingredients
# ---------------------------------------------------------------------------

# Keyword sets for ingredient classification
_FLOUR_KEYWORDS = {
    "flour",
    "rye",
    "wheat",
    "spelt",
    "emmer",
    "einkorn",
    "whole wheat",
    "wholemeal",
    "bread flour",
    "all-purpose",
    "all purpose",
    "white flour",
    "dark flour",
}
_STARTER_KEYWORDS = {
    "starter",
    "levain",
    "pre-ferment",
    "preferment",
    "sourdough starter",
    "leaven",
    "poolish",
    "biga",
}
_SALT_KEYWORDS = {"salt"}
_WATER_KEYWORDS = {"water", "liquid", "milk", "buttermilk"}


def _classify_ingredient(name: str) -> str:
    """Map a freeform ingredient name to one of: flour | starter | salt | water | other."""
    n = name.lower().strip()
    if any(k in n for k in _SALT_KEYWORDS):
        return "salt"
    if any(k in n for k in _STARTER_KEYWORDS):
        return "starter"
    if any(k in n for k in _FLOUR_KEYWORDS):
        return "flour"
    if any(k in n for k in _WATER_KEYWORDS):
        return "water"
    return "other"


def _parse_grams(amount_str: str) -> float | None:
    """Extract a gram value from a freeform amount string like '100g', '100 g', '0.5 kg'.

    Returns None if no numeric value can be parsed or if the unit is clearly non-gram
    (e.g. 'tsp', 'tbsp', 'cups' — volume units that can't be reliably converted).
    """
    if not amount_str:
        return None
    s = amount_str.lower().strip()

    # Reject obvious volume units that cannot be reliably converted to grams
    if re.search(r"(tsp|tbsp|cup|cups|ml|l\b|oz|lb|lbs|litre|liter)", s):
        return None

    # Handle kg → g
    kg_match = re.search(r"([\d.]+)\s*kg", s)
    if kg_match:
        try:
            return float(kg_match.group(1)) * 1000
        except ValueError:
            return None

    # Extract first numeric value (handles "300g", "300 g", "300", "~300g")
    match = re.search(r"[\d.]+", s)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


def _parse_baker_pcts_from_ingredients(
    ingredients: list[dict],
) -> dict[str, float | None]:
    """Compute baker's percentages for starter and salt from extracted ingredient list.

    Ingredient dicts have the shape: {"name": str, "amount": str}
    Amount strings are freeform (e.g. "300g", "100 g", "0.5 kg").

    Returns:
        {"starter_pct": float | None, "salt_pct": float | None}

    Returns None for a value when:
    - The relevant ingredient is absent from the list
    - Total flour weight is zero or cannot be parsed
    - The gram amount cannot be reliably parsed (volume units, etc.)
    """
    totals: dict[str, float] = {"flour": 0.0, "starter": 0.0, "salt": 0.0, "water": 0.0}

    for ing in ingredients:
        category = _classify_ingredient(ing.get("name", ""))
        if category not in totals:
            continue
        grams = _parse_grams(ing.get("amount", ""))
        if grams is not None:
            totals[category] += grams

    flour_g = totals["flour"]
    if flour_g <= 0:
        logger.info(
            "[BakerPcts] Cannot compute percentages: total flour weight is 0 or unparseable"
        )
        return {"starter_pct": None, "salt_pct": None}

    result: dict[str, float | None] = {}

    if totals["starter"] > 0:
        result["starter_pct"] = round(totals["starter"] / flour_g * 100, 1)
    else:
        result["starter_pct"] = None

    if totals["salt"] > 0:
        result["salt_pct"] = round(totals["salt"] / flour_g * 100, 1)
    else:
        result["salt_pct"] = None

    logger.info(
        f"[BakerPcts] flour={flour_g}g → "
        f"starter_pct={result['starter_pct']}, salt_pct={result['salt_pct']}"
    )
    return result


# Timeline builder
# ---------------------------------------------------------------------------


def build_timeline(state: SourdoughState) -> dict:
    """Build a baking timeline from parameters.

    For country_loaf: uses the hardcoded deterministic template.
    For all other products: extracts steps + ingredients from KB docs via LLM,
    then computes timestamps deterministically with calculate_timeline().
    """
    params = state.get("intent_params", {})

    # Resolve bread type
    raw_product = params.get("target_product", "")
    product_type = normalize_product_type(raw_product or "")
    # custom_product: True whenever the user asked for something other than country loaf.
    # Must be computed BEFORE defaulting product_type, so that unrecognized products
    # (e.g. "focaccia", "cinnamon rolls") are correctly treated as custom even though
    # product_type falls back to "country_loaf" for config purposes.
    custom_product = bool(raw_product) and product_type != "country_loaf"
    if product_type is None:
        product_type = "country_loaf"
    bread_config = BREAD_TYPES.get(product_type, BREAD_TYPES["country_loaf"])

    temp_c = max(3.0, min(safe_float(params.get("temperature_c"), 24), 40.0))
    hydration = max(
        50.0,
        min(
            safe_float(params.get("hydration"), bread_config["default_hydration"]),
            100.0,
        ),
    )
    starter_pct = max(
        5.0,
        min(
            safe_float(params.get("starter_pct"), bread_config["default_starter_pct"]),
            50.0,
        ),
    )
    num_loaves = max(1, min(safe_int(params.get("num_loaves"), 1), 20))
    flour_g = max(
        100.0,
        safe_float(
            params.get("flour_g"), num_loaves * bread_config["flour_per_unit_g"]
        ),
    )

    raw_bulk = estimate_fermentation_time(temp_c, hydration, starter_pct)
    bulk_hours = round(raw_bulk * bread_config["fermentation_factor"], 1)

    product_display_name = (
        bread_config["display_name"]
        if not custom_product
        else (raw_product or "").strip()
    )

    # -----------------------------------------------------------------------
    # Determine steps, timeline_source, and extracted ingredients
    # -----------------------------------------------------------------------
    timeline_source: str
    extracted_ingredients: list[dict] = []
    retrieved_docs = state.get("retrieved_docs", [])
    _rag_salt_pct: float | None = (
        None  # set in the extraction branch if RAG provides it
    )

    if not custom_product:
        # Country loaf — hardcoded template, no LLM call needed
        steps = default_bread_steps(bulk_hours, num_loaves)
        timeline_source = "hardcoded"
        logger.info("[Timeline] country_loaf → hardcoded steps")

    elif not retrieved_docs:
        # No KB results at all — cannot build a real plan
        steps = default_bread_steps(bulk_hours, num_loaves)
        timeline_source = "no_docs"
        logger.info(f"[Timeline] No docs retrieved for '{product_display_name}'")

    else:
        # Non-country-loaf with KB docs — extract real steps + ingredients from recipe
        extracted = _extract_recipe_from_docs(
            retrieved_docs, product_display_name, bulk_hours, num_loaves
        )

        if extracted is None:
            # LLM/parse failure — fall back gracefully, don't crash
            steps = default_bread_steps(bulk_hours, num_loaves)
            timeline_source = "no_recipe_found"
            logger.warning(
                f"[Timeline] Extraction returned None for '{product_display_name}' — using fallback"
            )

        elif not extracted["recipe_found"] or not extracted["steps"]:
            # Docs found but no matching recipe inside them
            steps = default_bread_steps(bulk_hours, num_loaves)
            timeline_source = "no_recipe_found"
            logger.info(
                f"[Timeline] No recipe for '{product_display_name}' found in docs"
            )

        else:
            steps = extracted["steps"]
            extracted_ingredients = extracted["ingredients"]
            timeline_source = "extracted"
            logger.info(
                f"[Timeline] Extracted {len(steps)} steps, "
                f"{len(extracted_ingredients)} ingredients for '{product_display_name}'"
            )

            # Back-fill starter_pct / salt_pct from RAG ingredients when the user
            # did not explicitly provide them.  User-supplied values always win.
            if extracted_ingredients:
                rag_pcts = _parse_baker_pcts_from_ingredients(extracted_ingredients)
                if rag_pcts["starter_pct"] is not None and "starter_pct" not in params:
                    starter_pct = max(5.0, min(rag_pcts["starter_pct"], 50.0))
                    logger.info(
                        f"[Timeline] Using RAG-derived starter_pct={starter_pct}% "
                        f"(user did not specify)"
                    )
                if rag_pcts["salt_pct"] is not None and "salt_pct" not in params:
                    # Store for use in compute_recipe below; keep a local var
                    _rag_salt_pct = rag_pcts["salt_pct"]
                    logger.info(
                        f"[Timeline] Using RAG-derived salt_pct={_rag_salt_pct}% "
                        f"(user did not specify)"
                    )
                else:
                    _rag_salt_pct = None

                # Annotate each extracted ingredient with its baker's percentage so
                # the presentation layer can render a proper three-column table.
                # Only flour, water, starter, and salt get a numeric %; everything
                # else (olive oil, toppings, seeds, etc.) gets "-".
                _flour_total_rag = sum(
                    (_parse_grams(ing["amount"]) or 0.0)
                    for ing in extracted_ingredients
                    if _classify_ingredient(ing["name"]) == "flour"
                )
                if _flour_total_rag > 0:
                    for ing in extracted_ingredients:
                        category = _classify_ingredient(ing["name"])
                        if category == "flour":
                            ing["baker_pct"] = "100%"
                        elif category in ("water", "starter", "salt"):
                            grams = _parse_grams(ing["amount"])
                            if grams is not None:
                                pct = round(grams / _flour_total_rag * 100, 1)
                                ing["baker_pct"] = f"{pct}%"
                            else:
                                ing["baker_pct"] = "-"
                        else:
                            ing["baker_pct"] = "-"
            else:
                _rag_salt_pct = None

            # Re-compute bulk fermentation with the (possibly updated) starter_pct
            raw_bulk = estimate_fermentation_time(temp_c, hydration, starter_pct)
            bulk_hours = round(raw_bulk * bread_config["fermentation_factor"], 1)

    # -----------------------------------------------------------------------
    # Compute timestamps deterministically
    # -----------------------------------------------------------------------
    constraints = {}
    start_time = None
    has_both = bool(params.get("start_time") and params.get("ready_by"))

    if params.get("start_time"):
        try:
            start_time = _parse_time(params["start_time"])
        except (ValueError, TypeError):
            logger.warning(
                f"[Timeline] Could not parse start_time '{params['start_time']}', using now"
            )
            start_time = datetime.now()
    elif params.get("ready_by"):
        constraints["ready_by"] = params["ready_by"]

    timeline = calculate_timeline(steps, start_time=start_time, constraints=constraints)

    # Ingredient baseline — only used for country_loaf (hardcoded path)
    # Priority: user param > RAG-derived > bread config default
    if "salt_pct" in params:
        resolved_salt_pct = safe_float(
            params["salt_pct"], bread_config["default_salt_pct"]
        )
    elif _rag_salt_pct is not None:
        resolved_salt_pct = _rag_salt_pct
    else:
        resolved_salt_pct = bread_config["default_salt_pct"]

    recipe = compute_recipe(
        product_type,
        flour_g,
        hydration_pct=hydration,
        starter_pct=starter_pct,
        salt_pct=resolved_salt_pct,
    )

    logger.info(
        f"[Timeline] {len(timeline)} steps, product={product_type}, "
        f"source={timeline_source}, bulk={bulk_hours}h, temp={temp_c}C, loaves={num_loaves}"
    )

    bake_plan_data: dict = {
        "timeline": timeline,
        "timeline_source": timeline_source,
        "product_type": product_type,
        "product_display_name": product_display_name,
        "num_loaves": num_loaves,
        "bulk_fermentation_hours": bulk_hours,
        "temperature_c": temp_c,
        "hydration": hydration,
        "recipe": recipe,  # baker's % baseline (country_loaf only)
        "extracted_ingredients": extracted_ingredients,  # KB-extracted ingredients (custom products)
        "custom_product": custom_product,
    }

    # Infeasibility check — both start_time AND ready_by provided
    if timeline and has_both:
        try:
            actual_finish = datetime.fromisoformat(timeline[-1]["end_time"])
            ready_dt = _parse_time(params["ready_by"])
            if actual_finish > ready_dt:
                total_minutes = sum(s["duration_minutes"] for s in steps)
                bake_plan_data["infeasible"] = True
                bake_plan_data["infeasible_reason"] = "both_constraints"
                bake_plan_data["infeasible_details"] = {
                    "start_time": start_time.strftime("%Y-%m-%d at %H:%M"),
                    "ready_by": ready_dt.strftime("%Y-%m-%d at %H:%M"),
                    "actual_finish": actual_finish.strftime("%Y-%m-%d at %H:%M"),
                    "total_hours": round(total_minutes / 60, 1),
                    "available_hours": round(
                        (ready_dt - start_time).total_seconds() / 3600, 1
                    ),
                }
                logger.warning(
                    f"[Timeline] Both constraints infeasible: starts {start_time.strftime('%H:%M')}, "
                    f"needs {round(total_minutes / 60, 1)}h but window closes {ready_dt.strftime('%H:%M')}"
                )
        except Exception as e:
            logger.warning(f"[Timeline] Both-constraints check failed: {e}")

    # Infeasibility check — only ready_by provided, working backwards
    elif timeline and not params.get("start_time") and params.get("ready_by"):
        now = datetime.now()
        plan_start = datetime.fromisoformat(timeline[0]["start_time"])
        if plan_start < now:
            total_minutes = sum(s["duration_minutes"] for s in steps)
            earliest_finish = now + timedelta(minutes=total_minutes)
            bake_plan_data["infeasible"] = True
            bake_plan_data["infeasible_reason"] = "past_start"
            bake_plan_data["infeasible_details"] = {
                "requested_ready_by": params.get("ready_by", ""),
                "required_start": plan_start.strftime("%Y-%m-%d at %H:%M"),
                "now": now.strftime("%Y-%m-%d at %H:%M"),
                "total_hours": round(total_minutes / 60, 1),
                "earliest_finish": earliest_finish.strftime("%Y-%m-%d at %H:%M"),
                "earliest_finish_iso": earliest_finish.isoformat(),
            }
            logger.warning(
                f"[Timeline] Infeasible: plan would start {plan_start.strftime('%H:%M')} "
                f"but it's already {now.strftime('%H:%M')}"
            )

    return {"bake_plan_data": bake_plan_data}


# ---------------------------------------------------------------------------
# Plan generator
# ---------------------------------------------------------------------------


def generate_bake_plan(state: SourdoughState) -> dict:
    """Generate a natural-language bake plan from the timeline."""
    plan_data = state.get("bake_plan_data", {})

    # --- Infeasibility: both start_time and ready_by given but window too short ---
    if (
        plan_data.get("infeasible")
        and plan_data.get("infeasible_reason") == "both_constraints"
    ):
        d = plan_data["infeasible_details"]
        answer = (
            f"**That window doesn't work** — your bake needs **{d['total_hours']} hours** but you've "
            f"only left **{d['available_hours']} hours** between "
            f"**{d['start_time']}** and **{d['ready_by']}**.\n\n"
            f"Please pick **one** constraint and I'll handle the other:\n\n"
            f"- **Give me a start time** → I'll calculate when the loaves will be ready.\n"
            f"  _(e.g. if you start at {d['start_time'].split(' at ')[-1]}, "
            f"they'll be ready at **{d['actual_finish'].split(' at ')[-1]}**)_\n\n"
            f"- **Give me a ready-by time** → I'll work out your start time and tell you "
            f"if it's still achievable.\n\n"
            f"Which would you prefer?"
        )
        logger.info(
            f"[BakePlan] Both-constraints infeasible: {d['available_hours']}h available, {d['total_hours']}h needed"
        )
        return {
            "response": answer,
            "steps": [
                {
                    "module": "bake_plan_infeasible",
                    "prompt": state["user_query"],
                    "response": answer,
                }
            ],
        }

    # --- Infeasibility: ready_by too soon (start would be in the past) ---
    if plan_data.get("infeasible"):
        d = plan_data["infeasible_details"]
        answer = (
            f"**That deadline isn't reachable — the bake would need to have started at "
            f"{d['required_start']}, which has already passed.**\n\n"
            f"Here's why: your bake requires **{d['total_hours']} hours** from start to finish. "
            f"Working backwards from your requested ready-by time puts the start in the past.\n\n"
            f"**What you can do:**\n"
            f"- **Start right now** — if you begin immediately, your loaves will be ready by "
            f"**{d['earliest_finish']}**.\n"
            f"- **Pick a later deadline** — tell me a new ready-by time and I'll build a fresh plan.\n\n"
            f"Would you like me to plan for **{d['earliest_finish']}** or a different time?"
        )
        logger.info(
            f"[BakePlan] Infeasibility notice (start was {d['required_start']})"
        )
        return {
            "response": answer,
            "steps": [
                {
                    "module": "bake_plan_infeasible",
                    "prompt": state["user_query"],
                    "response": answer,
                }
            ],
        }

    timeline_source = plan_data.get("timeline_source", "hardcoded")
    product_name = plan_data.get("product_display_name", "Sourdough")

    # --- No recipe in KB — decline and suggest alternatives ---
    if timeline_source in ("no_docs", "no_recipe_found"):
        answer = (
            f"I don't have a **{product_name}** recipe in my knowledge base, "
            f"so I can't build a reliable bake plan for it.\n\n"
            f"My knowledge base focuses on sourdough breads and techniques. "
            f"**Country Loaf** is the one type I always have a complete, verified plan for. "
            f"For other sourdough breads (focaccia, rye, whole wheat, baguettes, etc.) I'll search my sources — "
            f"results depend on what's in the knowledge base.\n\n"
            f"Would you like a **Country Loaf** plan, or try another sourdough bread?"
        )
        logger.info(
            f"[BakePlan] '{product_name}' — {timeline_source}, insufficient KB data"
        )
        return {
            "response": answer,
            "steps": [
                {
                    "module": "bake_plan_unsupported",
                    "prompt": state["user_query"],
                    "response": answer,
                }
            ],
            "bake_plan_data": {},
        }

    # --- Build LLM prompt ---
    llm = get_llm()
    timeline = plan_data.get("timeline", [])

    # Context docs for tips / sensory cues
    all_docs = state.get("retrieved_docs", [])
    context = "\n\n".join(
        f"[{doc.get('source', '?')}]: {doc.get('text', '')}" for doc in all_docs[:5]
    )

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for msg in state.get("messages", [])[-HISTORY_WINDOW:]:
        role = getattr(msg, "type", "user")
        content = getattr(msg, "content", str(msg))
        if role == "human":
            messages.append(HumanMessage(content=content))
        elif role == "ai":
            messages.append(AIMessage(content=content))

    if timeline_source == "extracted":
        # Steps and ingredients came from the actual KB recipe — present them directly.
        extracted_ingredients = plan_data.get("extracted_ingredients", [])
        num_units = plan_data.get("num_loaves", 1)

        if extracted_ingredients:
            ingredient_rows = "\n".join(
                f"| {ing['name']} | {ing['amount']} | {ing.get('baker_pct', '-')} |"
                for ing in extracted_ingredients
            )
            ingredient_block = (
                f"| Ingredient | Amount (× {num_units} unit(s)) | Baker's % |\n"
                f"|---|---|---|\n"
                f"{ingredient_rows}"
            )
        else:
            ingredient_block = (
                "_Ingredient quantities not extracted — see source documents._"
            )

        user_prompt = f"""Context from knowledge base (recipe source):
{context}

Product: {product_name}
Number of units: {num_units}
Kitchen temperature: {plan_data.get("temperature_c", "?")}°C

## Ingredients (extracted from the recipe, scaled to {num_units} unit(s)):
{ingredient_block}

## Baking timeline (steps from the recipe, timestamps computed):
{json.dumps(timeline, indent=2)}

User request: {state["user_query"]}

Present a detailed, friendly bake plan using the ingredients and timeline above exactly as given.
Use the context documents for sensory cues, temperatures, and tips — cite the source for each tip.
Do NOT substitute or invent ingredient quantities. Do NOT add country-loaf steps (no Dutch oven, no bannetons, no scoring) unless the recipe above explicitly calls for them.
When presenting the ingredients table, preserve all three columns (Ingredient, Amount, Baker's %) exactly as given above."""

    else:
        # Country loaf — computed recipe and hardcoded timeline
        recipe = plan_data.get("recipe", {})
        recipe_lines = [
            f"- Flour: **{recipe.get('flour_g', '?')}g** (hydration {recipe.get('hydration_pct', '?')}%)",
            f"- Water: **{recipe.get('water_g', '?')}g**",
            f"- Starter (levain): **{recipe.get('starter_g', '?')}g** ({recipe.get('starter_pct', '?')}%)",
            f"- Salt: **{recipe.get('salt_g', '?')}g** ({recipe.get('salt_pct', '?')}%)",
        ]
        if recipe.get("extras"):
            for k, v in recipe["extras"].items():
                recipe_lines.append(f"- {k.replace('_', ' ').title()}: **{v}g**")
        recipe_lines.append(f"- **Total dough: {recipe.get('total_dough_g', '?')}g**")
        if recipe.get("flour_note"):
            recipe_lines.append(f"- _{recipe['flour_note']}_")
        recipe_block = "\n".join(recipe_lines)

        user_prompt = f"""Context from knowledge base:
{context}

Product: {product_name}
Number of loaves: {plan_data.get("num_loaves", 1)}
Kitchen temperature: {plan_data.get("temperature_c", "?")}°C
Bulk fermentation: {plan_data.get("bulk_fermentation_hours", "?")} hours

## Ingredient weights (computed from baker's percentages):
{recipe_block}

## Baking timeline (computed):
{json.dumps(timeline, indent=2)}

User request: {state["user_query"]}

Create a detailed, friendly bake plan with ## Ingredients, ## Timeline, ## Tips, ## Sources:"""

    messages.append(HumanMessage(content=user_prompt))

    logger.info(
        f"[BakePlan] Generating plan for: {state['user_query']} (source={timeline_source})"
    )
    response = llm.invoke(messages)
    answer = response.content.strip()

    logger.info(f"[BakePlan] Response length: {len(answer)} chars")
    if not answer:
        logger.warning("[BakePlan] Empty answer!")

    step = {"module": "bake_plan", "prompt": user_prompt, "response": answer}
    return {"response": answer, "steps": [step]}


# ---------------------------------------------------------------------------
# Session store
# ---------------------------------------------------------------------------


def store_bake_session(state: SourdoughState) -> dict:
    """Save the bake plan to Supabase for polling notifications."""
    from app.tools.bake_session import save_bake_plan

    plan_data = state.get("bake_plan_data", {})
    session_id = state.get("session_id", "")

    if plan_data.get("infeasible"):
        logger.info("[StoreBakeSession] Skipping save — plan is infeasible")
    return {}
    # end of module

    if plan_data and session_id:
        try:
            save_bake_plan(session_id, plan_data)
            logger.info(f"[StoreBakeSession] Saved plan for session {session_id}")
        except Exception as e:
            logger.warning(f"[StoreBakeSession] Failed to save: {e}")

    return {}
