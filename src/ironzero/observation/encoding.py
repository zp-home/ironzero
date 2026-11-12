# SPDX-License-Identifier: Apache-2.0
"""The observation encoder: raw payload dict -> flat ``float32`` vector.

Block layout (offsets are cumulative, exactly as ``MicroDuelConfig`` declares):

1. ``global_dim``   — clock, army sizes, HP, centroids, funds/income, extractors;
2. ``max_entities * entity_features`` — own entities, slot-indexed;
3. ``max_entities * entity_features`` — visible enemies, slot-indexed;
4. ``map_channels * map_grid * map_grid`` — coarse control grid:
   ``[own count, enemy count, own - enemy, terrain passability]``.

Blocks 1 and 4 are the only ones with normalisation; the per-entity features come
from :mod:`ironzero.observation.features` and the terrain prior from
:mod:`ironzero.observation.terrain_channel`.
"""

from __future__ import annotations

import numpy as np

from ..micro_config import DEFAULT_CONFIG, MicroDuelConfig
from .features import _cell, _entity_features, _nearest
from .terrain_channel import terrain_passable_channel

__all__ = ["encode_observation"]


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
    # --- economy block ---------------------------------------------------
    # Without these three features the policy is asked to choose build
    # actions while being blind to its own bank balance. The mock engine and
    # the real bridge both report ``funds``/``income`` in ``global_state``.
    gstate = state.get("global_state", {}) or {}
    funds = float(gstate.get("funds") or 0.0)
    income = float(gstate.get("income") or 0.0)
    extractor_slots = float(sum(1 for s in (state.get("resource_slots") or []) if s.get("occupied")))
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
            min(max(funds / max(1.0, cfg.funds_scale), 0.0), 1.0),
            min(max(income / max(1.0, cfg.income_scale), 0.0), 1.0),
            min(extractor_slots / max(1.0, float(cfg.resource_slots_count)), 1.0),
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
