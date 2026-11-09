# SPDX-License-Identifier: Apache-2.0
"""Observation encoding and action masks for the micro-duel environment.

The encoder turns the raw observation payload produced by the game-side
adapter (mock engine or a future RWPP bridge) into a flat ``float32`` vector,
and derives the per-head action masks used by the policy. Keeping both in one
module guarantees the mask layout and the observation layout stay aligned with
``MicroDuelConfig``.
"""

from __future__ import annotations

import math
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np

from .micro_config import DEFAULT_CONFIG, MicroDuelConfig
from .mock_engine import ORDER_IDS
from .terrain import TerrainGrid

GROUP_DIM = 4  # matches the four unit groups in the micro-duel scenario
REPO_ROOT = Path(__file__).resolve().parents[2]


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
    # Channel 3 is the static terrain passability prior: 1 = fully passable,
    # 0 = fully blocked (water/cliff/trees/lava on the real map). It is scaled
    # to the same map_grid as the army channels so a convolution sees both.
    # Channels 0..2 count units, so only they are normalised by army size.
    if cfg.map_channels > 3:
        terrain = terrain_passable_channel(cfg)
        if terrain is not None:
            grid[3] = terrain
    army_channels = min(3, cfg.map_channels)
    grid[:army_channels] /= max(1.0, cfg.units_per_team)
    vector[offset : offset + grid.size] = grid.reshape(-1)
    offset += grid.size

    assert offset == cfg.flat_obs_dim
    return vector


# ----------------------------------------------------------------------
# Terrain observation channel
# ----------------------------------------------------------------------
def terrain_passable_channel(config: MicroDuelConfig = DEFAULT_CONFIG) -> np.ndarray | None:
    """Downsampled terrain passability prior ``(map_grid, map_grid)`` in ``[0, 1]``.

    Returns ``None`` when the config has no terrain channel or the terrain file
    cannot be read (callers then leave the channel zeroed; the observation
    dimension is unaffected because it was fixed when the config was built).
    """

    if not config.terrain_channel:
        return None
    path = _resolve_terrain_path(config.terrain_path)
    if path is None:
        warnings.warn(
            f"terrain file not found: {config.terrain_path!r}; the terrain observation "
            "channel stays zero",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    channel = _terrain_passable_channel(str(path), int(config.map_grid))
    if channel is None:
        warnings.warn(
            f"terrain file could not be parsed: {path}; the terrain observation "
            "channel stays zero",
            RuntimeWarning,
            stacklevel=2,
        )
    return channel


def _resolve_terrain_path(path: str | None) -> Path | None:
    """Resolve a terrain path against cwd first, then the repository root."""

    if not path:
        return None
    candidate = Path(path)
    if candidate.is_file():
        return candidate.resolve()
    if not candidate.is_absolute():
        fallback = (REPO_ROOT / candidate).resolve()
        if fallback.is_file():
            return fallback
    return None


@lru_cache(maxsize=16)
def _terrain_passable_channel(path: str, resolution: int) -> np.ndarray | None:
    try:
        grid = TerrainGrid.from_file(path)
    except (OSError, ValueError):
        return None
    return _downsample_passable(grid, resolution)


def _downsample_passable(grid: TerrainGrid, resolution: int) -> np.ndarray:
    """Average the blocked mask into a ``resolution x resolution`` passable grid.

    Each output cell covers the same *fraction* of the map regardless of the
    terrain tile size, so this works for any map resolution.
    """

    resolution = max(1, int(resolution))
    out = np.ones((resolution, resolution), dtype=np.float32)
    if grid.width <= 0 or grid.height <= 0:
        return out
    for cell_y in range(resolution):
        y0 = grid.height_px * cell_y / resolution
        y1 = grid.height_px * (cell_y + 1) / resolution
        ty0 = min(grid.height - 1, max(0, int(math.floor(y0 / grid.tile_height))))
        ty1 = min(grid.height - 1, max(ty0, int(math.ceil(y1 / grid.tile_height)) - 1))
        for cell_x in range(resolution):
            x0 = grid.width_px * cell_x / resolution
            x1 = grid.width_px * (cell_x + 1) / resolution
            tx0 = min(grid.width - 1, max(0, int(math.floor(x0 / grid.tile_width))))
            tx1 = min(grid.width - 1, max(tx0, int(math.ceil(x1 / grid.tile_width)) - 1))
            total = 0
            passable = 0
            for ty in range(ty0, ty1 + 1):
                base = ty * grid.width
                for tx in range(tx0, tx1 + 1):
                    total += 1
                    if not grid.blocked[base + tx]:
                        passable += 1
            out[cell_y, cell_x] = (passable / total) if total else 1.0
    return out


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
    group_ids_known = any(unit.get("group_id") is not None for unit in own)
    if group_ids_known:
        for unit in own:
            group_mask[int(unit.get("group_id") or 0) % cfg.groups] = True
    elif own:
        # The real RWPP bridge streams unit ids but no ``group_id``: the
        # game-side controller derives groups from ``unit.id % groups``, so all
        # groups are addressable and masking the head down to group 0 would
        # silently disable 75% of the army.
        group_mask[: cfg.groups] = True
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
