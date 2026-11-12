# SPDX-License-Identifier: Apache-2.0
"""Per-entity and per-cell feature extraction for the observation vector.

Pure functions over one observation payload dictionary. They own no state and
never touch the encoder's output buffer: ``encode_observation`` decides *where*
the features land, this module decides *what* they are.
"""

from __future__ import annotations

import numpy as np

from ..micro_config import MicroDuelConfig
from ..mock_engine import ORDER_IDS

GROUP_DIM = 4  # matches the four unit groups in the micro-duel scenario

__all__ = ["GROUP_DIM", "_cell", "_entity_features", "_nearest", "_unit_group"]


def _entity_features(
    unit: dict,
    *,
    config: MicroDuelConfig,
    centroid: tuple[float, float],
    nearest: dict | None,
    enemy: bool,
) -> np.ndarray:
    cfg = config
    x = float(unit.get("x", 0.0))
    y = float(unit.get("y", 0.0))
    hp = float(unit.get("hp") or 0.0)
    max_hp = float(unit.get("max_hp") or cfg.unit_hp)
    order = ORDER_IDS.get(str(unit.get("order_type", "idle")), 0)
    group = _unit_group(unit, cfg)
    rel_x = (x - centroid[0]) / cfg.map_width
    rel_y = (y - centroid[1]) / cfg.map_height
    if enemy:
        nearest_dx = (x - centroid[0]) / cfg.map_width
        nearest_dy = (y - centroid[1]) / cfg.map_height
        distance = float(np.hypot(x - centroid[0], y - centroid[1])) / 1000.0
    else:
        if nearest is not None:
            nearest_dx = (float(nearest.get("x", 0.0)) - x) / cfg.map_width
            nearest_dy = (float(nearest.get("y", 0.0)) - y) / cfg.map_height
            distance = float(np.hypot(float(nearest.get("x", 0.0)) - x, float(nearest.get("y", 0.0)) - y)) / 1000.0
        else:
            nearest_dx = nearest_dy = distance = 0.0
    return np.array(
        [
            1.0,  # occupancy mask
            x / cfg.map_width,
            y / cfg.map_height,
            hp / max(1.0, max_hp),
            group / max(1, cfg.groups - 1),
            order / max(1, len(ORDER_IDS) - 1),
            rel_x,
            rel_y,
            nearest_dx,
            min(distance, 1.5) / 1.5,
        ],
        dtype=np.float32,
    )


def _unit_group(unit: dict, cfg: MicroDuelConfig) -> float:
    """Unit group index, inferred from ``entity_id % groups`` when not streamed.

    The real RWPP bridge does not send ``group_id``; the game-side controller
    derives groups from ``unit.id % groups`` (the same convention the action
    group head uses). Inferring it here keeps the entity block consistent with
    the action semantics instead of collapsing every unit to group 0.
    """

    raw = unit.get("group_id")
    if raw is not None:
        try:
            return float(int(raw))
        except (TypeError, ValueError):
            pass
    try:
        entity_id = int(unit.get("entity_id", 0))
    except (TypeError, ValueError):
        entity_id = 0
    return float(entity_id % max(1, cfg.groups))


def _cell(unit: dict, cfg: MicroDuelConfig) -> tuple[int, int]:
    x = float(unit.get("x", 0.0))
    y = float(unit.get("y", 0.0))
    gx = int(x / max(1.0, cfg.map_width) * cfg.map_grid)
    gy = int(y / max(1.0, cfg.map_height) * cfg.map_grid)
    return min(cfg.map_grid - 1, max(0, gx)), min(cfg.map_grid - 1, max(0, gy))


def _nearest(entities: list[dict], point: tuple[float, float]) -> dict | None:
    if not entities:
        return None
    return min(
        entities,
        key=lambda u: (float(u.get("x", 0.0)) - point[0]) ** 2 + (float(u.get("y", 0.0)) - point[1]) ** 2,
    )
