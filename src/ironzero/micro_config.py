# SPDX-License-Identifier: Apache-2.0
"""Configuration for the deterministic micro-duel reference environment.

The mock engine exists so the whole IronZero loop (bridge protocol ->
observation encoding -> action masks -> PPO -> evaluation -> replay) can be
built and tested without the real game client. Every constant here is a
contract shared by the engine, the encoders and the training code.
"""

from __future__ import annotations

from dataclasses import dataclass


def _grid_index_to_world(index: int, grid: int, size: float, margin: float) -> float:
    if grid <= 1:
        return size / 2.0
    span = size - 2.0 * margin
    return margin + span * index / (grid - 1)


@dataclass(frozen=True, slots=True)
class MicroDuelConfig:
    """Fixed micro-duel scenario: 8 tanks per side, no economy, no fog.

    Defaults are calibrated from Rusted Warfare 1.15 ``assets/units/tanks/tank.ini``
    (HP 210, direct damage 25, range 130, shoot delay 75 ticks, move speed 1.1
    units/tick at a 60 Hz simulation loop) and a 1600x1600 fixed micro arena.
    """

    # Simulation clock (Rusted Warfare's fixed loop runs at 60 ticks/s)
    tick_rate: int = 60
    decision_interval: int = 30  # one model decision every 0.5 s
    episode_ticks: int = 3600  # 60 s episode

    # Map (small symmetric training arena; the real Lake map is 2600x2600)
    map_width: float = 1000.0
    map_height: float = 1000.0
    spawn_margin: float = 220.0

    # Units (real 1.15 tank values)
    teams: int = 2
    units_per_team: int = 8
    groups: int = 4
    unit_hp: float = 210.0
    unit_damage: float = 25.0
    attack_range: float = 130.0
    attack_cooldown_ticks: int = 75
    unit_speed: float = 66.0  # world units per second (1.1 units/tick * 60)
    unit_radius: float = 11.0
    unit_price: float = 350.0

    # Fog of war: 0 disables it (default). When > 0, enemies farther than this
    # distance from every friendly unit are hidden from observations/targeting.
    vision_radius: float = 0.0

    # Observation / action shape
    max_entities: int = 8
    target_slots: int = 9  # 8 enemy slots + "no target"
    position_grid: int = 8
    map_channels: int = 3
    map_grid: int = 8
    # Global feature block. The last three entries carry the economy state the
    # policy needs to decide *whether* it can afford a build action:
    #   [funds / funds_scale, income / income_scale, extractor_slots_owned]
    # They are appended to the original 12 so the layout stays a superset.
    global_dim: int = 15
    entity_features: int = 10
    # Per-entity *identity* features, written as a **separate block after the
    # control grid** rather than inside ``entity_features``.
    #
    # The ten ``entity_features`` are geometric (position / hp fraction / order /
    # distance) and contain no type information at all, so two completely
    # different situations encode to a bit-identical vector. Measured (see
    # ``runs/_scratch/probe_obs_type_blind.py``): "commandCenter 4000HP + builder
    # 170HP" and "tank 4000HP + scout 170HP" both produce the *same* 367-float
    # observation, max per-element difference 0.0. The policy therefore could not
    # tell a building from a vehicle, a factory from an extractor, or a builder
    # from a tank -- which is why "build something" was never learnable.
    #
    # Layout choice matters for checkpoints: appending the block at the tail
    # means the global / own / enemy / grid columns keep their exact previous
    # offsets, so a checkpoint saved before this block existed keeps its meaning
    # and only the new tail starts from zero influence. Inserting the new
    # features inside each entity's stride would have shifted every later column
    # and silently scrambled an old checkpoint.
    #
    # NOTE: that argument covers *this* block only. ``global_dim`` grew 12 -> 15
    # earlier and it sits at the *head*, so every column after it moved by three.
    # Migration therefore cannot be a plain "copy the leading columns" prefix
    # (see ``training.ppo._column_source``): columns are re-mapped block by block
    # so each weight stays attached to the feature it was trained on.
    #
    # 7 -> 9: two attack capabilities appended at the *tail* of each entity's
    # stride, so the first seven columns of every slot keep their exact meaning
    # and an older checkpoint's weights stay attached to the features they were
    # trained on. Before this the policy could describe where an entity was and
    # how much hp it had, but not whether it could shoot -- observed as a builder
    # walking straight into the enemy base to die. Columns 7/8 are
    # ``can_attack`` and ``attack_range / ATTACK_RANGE_SCALE``.
    type_features: int = 9
    # Fixed module-level denominator for the ``attack_range`` column (mirrored in
    # ``observation.features.ATTACK_RANGE_SCALE``). It is deliberately *not* a
    # config field: the map width changes per scenario, and a scenario-dependent
    # denominator would make the same tank read differently on Lake than on the
    # micro arena. 1000 px spans the whole stock 1.15 range ladder (tank 130,
    # artillery 290, turrets < 400); longer ranges clamp to 1.0 rather than break
    # the block's [0, 1] contract.
    # Normalisation denominators for the economy features. Fixed constants (not
    # derived from the scenario) so a checkpoint stays valid across maps.
    funds_scale: float = 5000.0
    income_scale: float = 40.0
    # Optional terrain passability layer. When set, ``map_channels`` should be 4
    # (three army channels + one static terrain channel) and the encoder reads
    # the blocked-tile grid from this JSON file (TerrainGrid ``map.json`` format).
    # ``None`` keeps the legacy 3-channel observation and the old ``obs_dim``.
    terrain_path: str | None = None

    # Reward shaping
    potential_coef: float = 0.5
    win_reward: float = 1.0
    loss_reward: float = -1.0

    # Economy and Macro extension
    enable_economy: bool = True
    starting_funds: float = 1000.0
    base_income_per_second: float = 1.0
    extractor_income_per_second: float = 8.0
    extractor_t2_income_per_second: float = 18.0
    extractor_cost: float = 700.0
    extractor_upgrade_cost: float = 1200.0
    factory_cost: float = 800.0
    turret_cost: float = 500.0
    tank_cost: float = 250.0
    resource_slots_count: int = 4

    # Economic reward & penalty weights
    extractor_reward: float = 0.5
    extractor_t2_reward: float = 0.8
    factory_reward: float = 0.6
    production_reward: float = 0.4
    building_loss_penalty: float = 1.0
    idle_funds_penalty_coef: float = 0.0001
    # Funds above this threshold are considered idle capital and are penalised.
    # Derived from the scenario's own price list rather than a magic number so
    # the penalty actually becomes reachable in short episodes.
    idle_funds_threshold: float = 800.0

    # Episode control
    opponent: str = "scripted"  # scripted | idle | random

    @property
    def action_nvec(self) -> tuple[int, int, int, int, int]:
        """Per-head action cardinalities, derived from the single source of truth.

        The ``kind`` head must stay in lock-step with ``mock_engine.ACTION_KINDS``
        (which defines the action id order) *and* with the Java bridge dispatch.
        ``ACTION_KINDS`` is imported lazily to avoid a circular import.
        """

        from .mock_engine import ACTION_KINDS

        return (
            len(ACTION_KINDS),  # kind: index into ACTION_KINDS
            self.groups + 1,  # per-group control + one broadcast slot
            self.position_grid,
            self.position_grid,
            self.target_slots,
        )

    @property
    def flat_obs_dim(self) -> int:
        entity_block = self.max_entities * self.entity_features
        grid_block = self.map_channels * self.map_grid * self.map_grid
        type_block = 2 * self.max_entities * self.type_features
        return self.global_dim + 2 * entity_block + grid_block + type_block

    @property
    def type_block_offset(self) -> int:
        """Start offset of the per-entity identity block (after the control grid)."""

        return (
            self.global_dim
            + 2 * self.max_entities * self.entity_features
            + self.map_channels * self.map_grid * self.map_grid
        )

    @property
    def terrain_channel(self) -> bool:
        """True when the observation carries the static terrain passability channel."""

        return self.map_channels > 3 and bool(self.terrain_path)

    def grid_to_world(self, gx: int, gy: int) -> tuple[float, float]:
        return (
            _grid_index_to_world(gx, self.position_grid, self.map_width, 24.0),
            _grid_index_to_world(gy, self.position_grid, self.map_height, 24.0),
        )


DEFAULT_CONFIG = MicroDuelConfig()
