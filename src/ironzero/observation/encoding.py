# SPDX-License-Identifier: Apache-2.0
"""The observation encoder: raw payload dict -> flat ``float32`` vector.

Block layout (offsets are cumulative, exactly as ``MicroDuelConfig`` declares):

1. ``global_dim``   鈥?clock, army sizes, HP, centroids, funds/income, extractors;
2. ``max_entities * entity_features`` 鈥?own entities, slot-indexed;
3. ``max_entities * entity_features`` 鈥?visible enemies, slot-indexed;
4. ``map_channels * map_grid * map_grid`` 鈥?coarse control grid:
   ``[own count, enemy count, own - enemy, terrain passability]``.

Blocks 1 and 4 are the only ones with normalisation; the per-entity features come
from :mod:`ironzero.observation.features` and the terrain prior from
:mod:`ironzero.observation.terrain_channel`.
"""

from __future__ import annotations

import numpy as np

from ..micro_config import DEFAULT_CONFIG, MicroDuelConfig
from .features import (
    _cell,
    _entity_features,
    _entity_type_features,
    _nearest,
    prioritise_own,
)
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
    # Normalise against the *actual* army value, not ``units_per_team *
    # unit_hp``. On a real economy map the army is buildings plus vehicles
    # (e.g. a 4000 HP command center and a 170 HP builder), so a 2-tank
    # denominator put own_hp at ~9.9 and enemy_hp at ~15.1 -- far outside the
    # [0, 1] range every other global feature is normalised to, which the
    # network sees as an out-of-distribution spike. Summing ``max_hp`` mirrors
    # the episode-start reference the reward path already uses
    # (``rwpp.reward.army_hp_reference``), and the configured army size stays
    # as the fallback for a payload that omits ``max_hp``.
    reference_hp = float(max(1, int(cfg.units_per_team)) * float(cfg.unit_hp))
    own_ref_hp = sum(
        float(u.get("max_hp") or cfg.unit_hp) for u in own
    ) or reference_hp
    enemy_ref_hp = sum(
        float(u.get("max_hp") or cfg.unit_hp) for u in enemies
    ) or reference_hp
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
            min(max(tick / max(1.0, float(cfg.episode_ticks)), 0.0), 1.0),
            # Army sizes are clamped: the AI reinforces beyond the configured
            # army size on economy maps, and an unclamped ratio would again
            # exceed the [0, 1] contract.
            min(max(len(own) / max(1.0, float(cfg.units_per_team)), 0.0), 1.0),
            min(max(len(enemies) / max(1.0, float(cfg.units_per_team)), 0.0), 1.0),
            min(max(own_hp / max(1.0, own_ref_hp), 0.0), 1.0),
            min(max(enemy_hp / max(1.0, enemy_ref_hp), 0.0), 1.0),
            min(max(own_cx / max(1.0, cfg.map_width), 0.0), 1.0),
            min(max(own_cy / max(1.0, cfg.map_height), 0.0), 1.0),
            min(max(enemy_cx / max(1.0, cfg.map_width), 0.0), 1.0),
            min(max(enemy_cy / max(1.0, cfg.map_height), 0.0), 1.0),
            # Directional deltas: these are *signed* by design and live in
            # [-1, 1], not [0, 1].
            max(-1.0, min(1.0, (enemy_cx - own_cx) / max(1.0, cfg.map_width))),
            max(-1.0, min(1.0, (enemy_cy - own_cy) / max(1.0, cfg.map_height))),
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
    # Slots are filled by *priority*, not in raw ``Unit.unitList`` order. On an
    # economy map the own side is a command center + builder + whatever it has
    # produced, and the stream order is spawn order, so with a small
    # ``max_entities`` the buildings that unlock every economic action are the
    # first thing to fall off the end. Buildings and builders therefore come
    # first; combat units fill the remaining slots. This keeps ``obs_dim`` and
    # the feature layout unchanged, so existing checkpoints still load.
    own_centroid = (own_cx, own_cy)
    nearest_enemy = _nearest(enemies, own_centroid)
    own_ordered = prioritise_own(own)
    for slot in range(cfg.max_entities):
        feats = np.zeros(cfg.entity_features, dtype=np.float32)
        if slot < len(own_ordered):
            unit = own_ordered[slot]
            feats = _entity_features(
                unit, config=cfg, centroid=own_centroid, nearest=nearest_enemy, enemy=False
            )
        vector[offset : offset + cfg.entity_features] = feats
        offset += cfg.entity_features

    # --- enemy entity block ----------------------------------------------
    # Deliberately *not* reordered: the target head indexes this list, and the
    # bridge streams entities in stable spawn order. Sorting by distance would
    # make "enemy slot i" change every frame as our own units move, so the
    # policy could not learn a consistent slot -> target mapping.
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

    # --- identity block (own then enemy, same slot order) -----------------
    # Written *after* the grid so every previously-existing column keeps its
    # offset: an older checkpoint reads exactly the same prefix it was trained
    # on and only the new tail starts at zero influence. See
    # ``MicroDuelConfig.type_features`` for why this block exists at all.
    own_identity = own_ordered
    enemy_identity = enemies
    for source in (own_identity, enemy_identity):
        for slot in range(cfg.max_entities):
            feats = np.zeros(cfg.type_features, dtype=np.float32)
            if slot < len(source):
                feats = _entity_type_features(source[slot])
            vector[offset : offset + cfg.type_features] = feats
            offset += cfg.type_features

    assert offset == cfg.flat_obs_dim
    return vector
