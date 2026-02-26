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
        # Extract the first valid number (handles "75%", "75 - 80%", "-5.5°C", etc.)
        matches = re.findall(r"-?\d+\.?\d*", value)
        if matches:
            try:
                return float(matches[0])
            except ValueError:
                return default
    return default


def safe_int(value, default: int) -> int:
    """Parse an int from an LLM-extracted value."""
    return int(safe_float(value, float(default)))
