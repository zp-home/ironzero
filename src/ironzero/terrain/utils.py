# SPDX-License-Identifier: Apache-2.0
"""Small numeric / validation helpers shared by the terrain subpackage.

Kept in its own module so ``grid.py``, ``passable.py`` and ``pathing.py`` can use
them without importing each other for helper access only.
"""

from __future__ import annotations

from typing import Any


def _positive_int(value: Any, field: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"terrain payload field {field!r} is not an int: {value!r}") from exc
    if number <= 0:
        raise ValueError(f"terrain payload field {field!r} must be positive, got {number}")
    return number


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def _clamp_int(value: int, low: int, high: int) -> int:
    return low if value < low else high if value > high else value


def _clamp_bounds(
    bounds: tuple[float, float, float, float] | None,
    width_px: int,
    height_px: int,
) -> tuple[float, float, float, float] | None:
    if bounds is None:
        return None
    min_x, min_y, max_x, max_y = (float(value) for value in bounds)
    min_x = _clamp(min_x, 0.0, float(width_px))
    max_x = _clamp(max_x, 0.0, float(width_px))
    min_y = _clamp(min_y, 0.0, float(height_px))
    max_y = _clamp(max_y, 0.0, float(height_px))
    if min_x > max_x:
        min_x, max_x = max_x, min_x
    if min_y > max_y:
        min_y, max_y = max_y, min_y
    return (min_x, min_y, max_x, max_y)
