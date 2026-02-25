"""Safe parameter parsing for values extracted by the LLM."""

import re


def safe_float(value, default: float) -> float:
    """Parse a float from an LLM-extracted value.

    Handles: 75, "75", "75%", "75.0%", None, "null", "None", "", lists, etc.
    """
    if value is None or value == "null" or value == "None":
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # Strip %, degrees, spaces, units
        cleaned = re.sub(r"[^\d.\-]", "", value)
        if cleaned:
            try:
                return float(cleaned)
            except ValueError:
                return default
    return default


def safe_int(value, default: int) -> int:
    """Parse an int from an LLM-extracted value."""
    return int(safe_float(value, float(default)))
