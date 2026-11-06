# SPDX-License-Identifier: Apache-2.0
"""Observation encoding and action masks for the micro-duel environment.

The encoder turns the raw observation payload produced by the game-side
adapter (mock engine or a future RWPP bridge) into a flat ``float32`` vector,
and derives the per-head action masks used by the policy. Keeping both in one
module guarantees the mask layout and the observation layout stay aligned with
``MicroDuelConfig``.
"""

from __future__ import annotations

import numpy as np

from .micro_config import DEFAULT_CONFIG, MicroDuelConfig
from .mock_engine import ORDER_IDS

GROUP_DIM = 4  # matches the four unit groups in the micro-duel scenario


def encode_observation(state: dict, config: MicroDuelConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Encode a raw observation payload into a flat float32 vector."""

    cfg = config
    vector = np.zeros(cfg.flat_obs_dim, dtype=np.float32)
    offset = 0

    # --- global block -----------------------------------------------------
    own = state.get("own_entities", [])
    enemies = state.get("visible_enemy_entities", [])
    global_state = state.get("global_state", {})
    tick = float(state.get("tick", 0))
    own_hp = sum(float(u.get("hp") or 0.0) for u in own)
    enemy_hp = sum(float(u.get("hp") or 0.0) for u in enemies)
    total_hp = float(cfg.units_per_team * cfg.unit_hp)
    own_cx = float(np.mean([float(u.get("x", 0.0)) for u in own])) if own else 0.0
    own_cy = float(np.mean([float(u.get("y", 0.0)) for u in own])) if own else 0.0
    enemy_cx = float(np.mean([float(u.get("x", 0.0)) for u in enemies])) if enemies else own_cx
    enemy_cy = float(np.mean([float(u.get("y", 0.0)) for u in enemies])) if enemies else own_cy
    g = np.array(
        [
            min(tick / max(1.0, float(cfg.episode_ticks)), 1.0),
            len(own) / cfg.units_per_team,
            len(enemies) / cfg.units_per_team,
            own_hp / total_hp,
            enemy_hp / total_hp,
            own_cx / cfg.map_width,
            own_cy / cfg.map_height,
            enemy_cx / cfg.map_width,
            enemy_cy / cfg.map_height,
            (enemy_cx - own_cx) / cfg.map_width,
            (enemy_cy - own_cy) / cfg.map_height,
            1.0,
        ],
        dtype=np.float32,
    )
    assert g.shape[0] == cfg.global_dim
    vector[offset : offset + cfg.global_dim] = g
    offset += cfg.global_dim

    # --- own entity block ------------------------------------------------
    own_centroid = (own_cx, own_cy)
    nearest_enemy = _nearest(enemies, own_centroid)
    for slot in range(cfg.max_entities):
        feats = np.zeros(cfg.entity_features, dtype=np.float32)
        if slot < len(own):
            unit = own[slot]
            feats = _entity_features(
                unit, config=cfg, centroid=own_centroid, nearest=nearest_enemy, enemy=False
            )
        vector[offset : offset + cfg.entity_features] = feats
        offset += cfg.entity_features

    # --- enemy entity block ----------------------------------------------
    for slot in range(cfg.max_entities):
        feats = np.zeros(cfg.entity_features, dtype=np.float32)
        if slot < len(enemies):
            feats = _entity_features(
                enemies[slot], config=cfg, centroid=own_centroid, nearest=None, enemy=True
            )
        vector[offset : offset + cfg.entity_features] = feats
        offset += cfg.entity_features

    # --- coarse control grid ---------------------------------------------
    grid = np.zeros((cfg.map_channels, cfg.map_grid, cfg.map_grid), dtype=np.float32)
    for unit in own:
        gx, gy = _cell(unit, cfg)
        grid[0, gy, gx] += 1.0
    for unit in enemies:
        gx, gy = _cell(unit, cfg)
        grid[1, gy, gx] += 1.0
    grid[2] = np.clip(grid[0] - grid[1], -1.0, 1.0)
    vector[offset : offset + grid.size] = grid.reshape(-1) / max(1.0, cfg.units_per_team)
    offset += grid.size

    assert offset == cfg.flat_obs_dim
    return vector


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
    group = float(unit.get("group_id", 0))
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


# ----------------------------------------------------------------------
# Action masks
# ----------------------------------------------------------------------
def build_masks(state: dict, config: MicroDuelConfig = DEFAULT_CONFIG) -> dict[str, np.ndarray]:
    """Return per-head masks.

    Layout matches ``MicroDuelConfig.action_nvec``:
    kind(5), group(G), x(P), y(P), target(T).
    """

    cfg = config
    own = state.get("own_entities", [])
    enemies = state.get("visible_enemy_entities", [])
    kind_mask = np.ones(5, dtype=bool)
    group_mask = np.zeros(cfg.groups + 1, dtype=bool)
    for unit in own:
        group_mask[int(unit.get("group_id", 0)) % cfg.groups] = True
    group_mask[-1] = True  # broadcast to every alive group
    x_mask = np.ones(cfg.position_grid, dtype=bool)
    y_mask = np.ones(cfg.position_grid, dtype=bool)
    target_mask = np.zeros(cfg.target_slots, dtype=bool)
    for slot in range(min(len(enemies), cfg.target_slots - 1)):
        target_mask[slot] = True
    target_mask[-1] = True  # "no target" fallback, keeps sampling valid
    return {
        "kind": kind_mask,
        "group": group_mask,
        "x": x_mask,
        "y": y_mask,
        "target": target_mask,
    }


def masks_as_lists(masks: dict[str, np.ndarray], config: MicroDuelConfig = DEFAULT_CONFIG) -> dict[str, list]:
    """Serialize masks for the ``ironzero/1`` observation message."""

    cfg = config
    position = np.outer(masks["x"], masks["y"])
    return {
        "action_mask": [bool(v) for v in masks["kind"]],
        "unit_group_mask": [bool(v) for v in masks["group"]],
        "target_mask": [bool(v) for v in masks["target"]],
        "position_mask": [[bool(v) for v in row] for row in position],
    }


def flat_mask_vector(masks: dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate masks into one boolean vector (debug / logging helper)."""

    return np.concatenate(
        [masks["kind"], masks["group"], masks["x"], masks["y"], masks["target"]]
    ).astype(bool)
