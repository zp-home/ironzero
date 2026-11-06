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

    # Observation / action shape
    max_entities: int = 8
    target_slots: int = 9  # 8 enemy slots + "no target"
    position_grid: int = 8
    map_channels: int = 3
    map_grid: int = 8
    global_dim: int = 12
    entity_features: int = 10

    # Reward shaping
    potential_coef: float = 0.5
    win_reward: float = 1.0
    loss_reward: float = -1.0

    # Episode control
    opponent: str = "scripted"  # scripted | idle | random

    @property
    def action_nvec(self) -> tuple[int, int, int, int, int]:
        return (
            5,  # kind: move, attack_move, attack_target, stop, hold
            self.groups + 1,  # per-group control + one broadcast slot
            self.position_grid,
            self.position_grid,
            self.target_slots,
        )

    @property
    def flat_obs_dim(self) -> int:
        entity_block = self.max_entities * self.entity_features
        grid_block = self.map_channels * self.map_grid * self.map_grid
        return self.global_dim + 2 * entity_block + grid_block

    def grid_to_world(self, gx: int, gy: int) -> tuple[float, float]:
        return (
            _grid_index_to_world(gx, self.position_grid, self.map_width, 24.0),
            _grid_index_to_world(gy, self.position_grid, self.map_height, 24.0),
        )


DEFAULT_CONFIG = MicroDuelConfig()
