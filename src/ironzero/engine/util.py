# SPDX-License-Identifier: Apache-2.0
"""Tiny numeric helper shared by the engine mixins."""

from __future__ import annotations


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value
