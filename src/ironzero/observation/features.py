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

__all__ = [
    "GROUP_DIM",
    "TYPE_ONE_HOT_NAMES",
    "_cell",
    "_entity_features",
    "_entity_type_features",
    "_nearest",
    "_unit_group",
    "prioritise_own",
]


def prioritise_own(units: list[dict]) -> list[dict]:
    """Order own entities so a small slot budget keeps the useful units.

    Buildings (factories first, since they produce units) and builders come
    before combat units: on an economy map they are what makes every non-combat
    action possible, and with ``max_entities`` slots the tail of the bridge
    stream is dropped. ``sorted`` is stable, so equal-ranked units keep their
    stream order and behaviour stays deterministic.

    Only the *own* block is reordered. The enemy block keeps the bridge's stream
    order because the target head indexes it: sorting enemies by distance would
    change which enemy slot *i* names as our own units move, and the policy
    could never learn a stable slot -> target mapping.
    """

    def rank(unit: dict) -> int:
        if unit.get("is_factory") or unit.get("is_factory_building"):
            return 0
        if unit.get("is_building"):
            return 1
        if unit.get("is_builder"):
            return 2
        return 3

    return sorted(units, key=rank)


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


#: Stable ordering of the unit-type id aliases written into the identity block.
#:
#: ``entity_type_id`` is a large, sparse integer (real bridge ids run into the
#: hundreds of thousands for unlisted types), so feeding it raw would dominate
#: the trunk input. It is therefore split into a small "is this one of the
#: economy types the action space actually talks about" one-hot plus a hashed
#: remainder that still distinguishes unknown types from each other.
TYPE_ONE_HOT_NAMES: tuple[str, ...] = ("tank", "builder", "extractor", "factory")


def _type_id(unit: dict) -> int:
    try:
        return int(unit.get("entity_type_id") or 0)
    except (TypeError, ValueError):
        return 0


def _entity_type_features(unit: dict) -> np.ndarray:
    """Identity features: what *kind* of thing this entity is.

    The geometric ``_entity_features`` block is type-blind by construction: it
    encodes position, hp fraction, order and distance only, so a command center
    and a tank of equal hp/max_hp/position produce identical numbers. That made
    "build a factory, then produce from it" unlearnable -- the policy could not
    observe that it owned a factory, that a builder was still alive, or that an
    enemy turret was a turret.

    Layout (``MicroDuelConfig.type_features`` = 7):

    ===== =========================================================
    0     occupancy (mirrors the geometric block, so a padded slot
          reads as "empty" here too)
    1     ``is_building``        -- 房屋 / 建筑
    2     ``is_factory``         -- 可以产兵的工厂
    3     ``is_builder``         -- 能建造的工程车
    4     ``is_air_unit``        -- 空中单位
    5     ``build_progress``     -- 在建进度 (1.0 = 已完工)
    6     hashed type id         -- 其余类型之间的区分
    ===== =========================================================

    Everything is in ``[0, 1]`` so this block obeys the same normalisation
    contract as the rest of the observation.
    """

    building = 1.0 if unit.get("is_building") else 0.0
    factory = 1.0 if (unit.get("is_factory") or unit.get("is_factory_building")) else 0.0
    builder = 1.0 if unit.get("is_builder") else 0.0
    air = 1.0 if unit.get("is_air_unit") else 0.0
    # A finished unit reports 1.0 (the bridge sends 1.0 for non-buildings); a
    # structure mid-construction reports its real fraction. Clamped so a stray
    # engine value cannot push the observation out of range.
    try:
        progress = float(unit.get("build_progress", 1.0))
    except (TypeError, ValueError):
        progress = 1.0
    if not np.isfinite(progress):
        progress = 1.0
    progress = float(min(1.0, max(0.0, progress)))
    # Hash the raw id into [0, 1). Deterministic across runs and platforms
    # (Python's int hash on a fixed seed is not, so use an explicit mix) and
    # monotone-free: we only need "different types look different".
    type_id = _type_id(unit)
    hashed = ((type_id * 2654435761) & 0xFFFFFFFF) / 0xFFFFFFFF
    return np.array(
        [1.0, building, factory, builder, air, progress, hashed],
        dtype=np.float32,
    )


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
