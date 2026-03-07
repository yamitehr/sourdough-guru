"""Deterministic baking math calculations — no LLM calls."""

import math
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


def calculate_hydration(water_g: float, flour_g: float) -> float:
    """Calculate hydration percentage."""
    if flour_g == 0:
        return 0.0
    return round((water_g / flour_g) * 100, 1)


def calculate_bakers_percentages(ingredients: dict[str, float], flour_g: float) -> dict[str, float]:
    """Calculate baker's percentages relative to total flour weight.

    ingredients: {"water": 750, "salt": 20, "starter": 200, ...}
    flour_g: total flour weight in grams
    """
    if flour_g == 0:
        return {}
    return {name: round((weight / flour_g) * 100, 1) for name, weight in ingredients.items()}


def scale_recipe(recipe: dict[str, float], target_flour_g: float, original_flour_g: float) -> dict[str, float]:
    """Scale a recipe to a target flour weight.

    recipe: {"flour": 1000, "water": 750, "salt": 20, ...}
    """
    if original_flour_g == 0:
        return recipe
    factor = target_flour_g / original_flour_g
    return {name: round(weight * factor, 1) for name, weight in recipe.items()}


def estimate_fermentation_time(
    temp_c: float,
    hydration_pct: float = 75.0,
    starter_pct: float = 20.0,
) -> float:
    """Estimate bulk fermentation time in hours.

    Based on temperature, hydration, and starter percentage.
    Higher temp / hydration / starter → faster fermentation.
    """

    # Base time at 24°C, 75% hydration, 20% starter
    base_hours = 4.0

    # Temperature factor: roughly halves/doubles every 8°C
    temp_factor = 2 ** ((24 - temp_c) / 8)

    # Hydration factor: higher hydration slightly speeds fermentation
    hydration_factor = 75 / max(hydration_pct, 50)

    # Starter factor: more starter = faster
    starter_factor = 20 / max(starter_pct, 5)

    hours = base_hours * temp_factor * hydration_factor * starter_factor
    return round(max(hours, 1.0), 1)


def _parse_time(time_str: str) -> datetime:
    """Parse a time string into a datetime, handling both ISO and casual formats."""
    # Try ISO format first
    try:
        dt = datetime.fromisoformat(time_str)
        # Strip timezone to keep all datetimes naive (consistent with the rest of the codebase)
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        return dt
    except ValueError:
        pass

    # Handle casual formats like "6am", "6:30pm", "14:00"
    time_str = time_str.strip().lower().replace(" ", "")
    match = re.match(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", time_str)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        period = match.group(3)
        if period == "pm" and hour < 12:
            hour += 12
        elif period == "am" and hour == 12:
            hour = 0

        now = datetime.now(ZoneInfo("Asia/Jerusalem")).replace(tzinfo=None)
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        # If the target time is in the past, assume tomorrow
        if target <= now:
            target += timedelta(days=1)
        return target

    raise ValueError(f"Cannot parse time: {time_str}")


def calculate_timeline(
    steps: list[dict],
    start_time: datetime | None = None,
    constraints: dict | None = None,
) -> list[dict]:
    """Build a baking timeline from a list of steps.

    Each step: {"name": str, "duration_minutes": int, "description": str}
    constraints: {"ready_by": datetime} to work backwards

    Returns steps with added "start_time" and "end_time" fields.
    """
    if not steps:
        return []

    total_minutes = sum(s["duration_minutes"] for s in steps)

    if constraints and "ready_by" in constraints:
        # Work backwards from the target time
        ready_by = constraints["ready_by"]
        if isinstance(ready_by, str):
            ready_by = _parse_time(ready_by)
        start_time = ready_by - timedelta(minutes=total_minutes)
    elif start_time is None:
        start_time = datetime.now(ZoneInfo("Asia/Jerusalem")).replace(tzinfo=None)

    timeline = []
    current = start_time
    for step in steps:
        end = current + timedelta(minutes=step["duration_minutes"])
        timeline.append({
            **step,
            "start_time": current.isoformat(),
            "end_time": end.isoformat(),
        })
        current = end

    return timeline


def default_bread_steps(
    bulk_hours: float,
    num_loaves: int = 1,
) -> list[dict]:
    """Generate default bread-making steps for a standard sourdough loaf."""
    bulk_minutes = int(bulk_hours * 60)
    return [
        {"name": "Mix & Autolyse", "duration_minutes": 40, "description": "Combine flour and water, rest."},
        {"name": "Add starter & salt", "duration_minutes": 10, "description": "Incorporate levain and salt, mix well."},
        {"name": "Bulk fermentation", "duration_minutes": bulk_minutes, "description": f"Bulk ferment with stretch & folds every 30 min for first 2 hours."},
        {"name": "Pre-shape", "duration_minutes": 15, "description": f"Divide dough into {num_loaves} piece(s), pre-shape into rounds."},
        {"name": "Bench rest", "duration_minutes": 20, "description": "Rest on bench for gluten relaxation."},
        {"name": "Final shape", "duration_minutes": 15, "description": "Shape into batards or boules, place in bannetons."},
        {"name": "Cold retard", "duration_minutes": 720, "description": "Refrigerate overnight (8-14 hours) for flavor development."},
        {"name": "Preheat oven", "duration_minutes": 60, "description": "Preheat oven to 500°F / 260°C with Dutch oven inside."},
        {"name": "Score & bake (covered)", "duration_minutes": 20, "description": "Score loaves, bake covered for steam."},
        {"name": "Bake (uncovered)", "duration_minutes": 25, "description": "Remove lid, bake until deep golden brown."},
        {"name": "Cool", "duration_minutes": 60, "description": "Cool on wire rack — resist cutting for at least 1 hour."},
    ]


# ---------------------------------------------------------------------------
# Bread-type registry
# ---------------------------------------------------------------------------

BREAD_TYPES: dict[str, dict] = {
    "country_loaf": {
        "display_name": "Country Loaf (Pain de Campagne)",
        "default_hydration": 75.0,
        "default_starter_pct": 20.0,
        "default_salt_pct": 2.0,
        "flour_per_unit_g": 500,
        "fermentation_factor": 1.0,
        "extras": {},
        "flour_note": None,
    },
}

_PRODUCT_ALIASES: dict[str, str] = {
    # country loaf (+ generic terms that map here)
    "country loaf": "country_loaf",
    "country_loaf": "country_loaf",
    "pain de campagne": "country_loaf",
    "sourdough loaf": "country_loaf",
    "basic loaf": "country_loaf",
    "standard loaf": "country_loaf",
    "white sourdough": "country_loaf",
    "sourdough": "country_loaf",
    "loaf": "country_loaf",
    "loaves": "country_loaf",
    "bread": "country_loaf",
    "white bread": "country_loaf",
    "sourdough bread": "country_loaf",
}


def normalize_product_type(target_product: str) -> str | None:
    """Normalize a product string to a known BREAD_TYPES key, or None if unrecognized.

    Tries exact match first, then strips common suffixes/prefixes and retries.
    """
    if not target_product:
        return None
    key = target_product.strip().lower().replace("-", " ")

    # Exact match
    result = _PRODUCT_ALIASES.get(key)
    if result:
        return result

    # Iteratively strip common suffixes and prefixes, retrying after each strip
    _SUFFIXES = [" loaves", " loaf", " bread", " sourdough"]
    _PREFIXES = ["sourdough "]
    changed = True
    while changed:
        changed = False
        for suffix in _SUFFIXES:
            if key.endswith(suffix):
                key = key[: -len(suffix)].strip()
                changed = True
                result = _PRODUCT_ALIASES.get(key)
                if result:
                    return result
        for prefix in _PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix) :].strip()
                changed = True
                result = _PRODUCT_ALIASES.get(key)
                if result:
                    return result

    return None


def compute_recipe(
    product_type: str,
    flour_g: float,
    hydration_pct: float | None = None,
    starter_pct: float | None = None,
    salt_pct: float | None = None,
) -> dict:
    """Compute ingredient weights (grams) from baker's percentages.

    Falls back to BREAD_TYPES defaults for any None parameter.
    """
    config = BREAD_TYPES.get(product_type, BREAD_TYPES["country_loaf"])
    h = hydration_pct if hydration_pct is not None else config["default_hydration"]
    sp = starter_pct if starter_pct is not None else config["default_starter_pct"]
    saltp = salt_pct if salt_pct is not None else config["default_salt_pct"]

    water_g = round(flour_g * h / 100, 1)
    starter_g = round(flour_g * sp / 100, 1)
    salt_g = round(flour_g * saltp / 100, 1)

    extras: dict[str, float] = {
        key: round(flour_g / config["flour_per_unit_g"] * per_unit, 1)
        for key, per_unit in config["extras"].items()
    }

    total_g = round(flour_g + water_g + starter_g + salt_g + sum(extras.values()), 1)

    recipe: dict = {
        "flour_g": flour_g,
        "water_g": water_g,
        "starter_g": starter_g,
        "salt_g": salt_g,
        "total_dough_g": total_g,
        "hydration_pct": h,
        "starter_pct": sp,
        "salt_pct": saltp,
    }
    if extras:
        recipe["extras"] = extras
    if config.get("flour_note"):
        recipe["flour_note"] = config["flour_note"]
    return recipe


