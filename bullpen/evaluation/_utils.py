"""Small helpers shared across the evaluation readers."""

from __future__ import annotations


def as_float_or_nan(value: object) -> float:
    """``value`` as a float, or ``NaN`` when it is absent or not a number."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
