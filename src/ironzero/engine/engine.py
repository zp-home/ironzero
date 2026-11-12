# SPDX-License-Identifier: Apache-2.0
"""MicroDuelEngine: the composed engine plus its orchestration entry points.

The engine itself is assembled from mixins (spawn, actions, movement, combat,
economy, opponents, observation, reward); this module keeps only the pieces
that tie them together -- ``__init__``, the tick loop, ``advance()``,
``step_decision()`` and the diagnostics (``snapshot`` / ``evaluate``).
"""

from __future__ import annotations

import random
from typing import Any, Sequence

from ..micro_config import DEFAULT_CONFIG, MicroDuelConfig
from ..terrain import TerrainGrid
from .actions import ActionMixin
from .combat import CombatMixin
from .config import make_terrain_config
from .economy import EconomyMixin
from .entities import Unit
from .movement import MovementMixin
from .observation import ObservationMixin
from .opponents import OpponentMixin
from .reward import RewardMixin
from .spawn import SpawnMixin


class MicroDuelEngine(
    SpawnMixin,
    ActionMixin,
    MovementMixin,
    CombatMixin,
    EconomyMixin,
    OpponentMixin,
    ObservationMixin,
    RewardMixin,
):
    """Two teams of tanks on an open map, micro control only."""

    def __init__(
        self,
        config: MicroDuelConfig | None = None,
        opponent: str | None = None,
        terrain: TerrainGrid | None = None,
    ) -> None:
        self.terrain = terrain
        base_config = config or DEFAULT_CONFIG
        if terrain is not None:
            # Terrain mode: the real map's pixel size replaces the arena size.
            self.config = make_terrain_config(terrain, base_config)
        else:
            self.config = base_config
        self.opponent = opponent or self.config.opponent
        if self.opponent not in {"scripted", "idle", "random"}:
            raise ValueError(f"unknown opponent: {self.opponent}")
        self._rng = random.Random(0)
        self.units: list[Unit] = []
        self.tick = 0
        self.done = False
        self.result: str | None = None
        self.events: list[dict[str, Any]] = []
        self.stats: dict[str, Any] = {}
        self._phi_prev = 0.0
        self._pending_reward = 0.0
        self.reset(seed=0)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def alive_units(self, team: int | None = None) -> list[Unit]:
        return [u for u in self.units if u.alive and (team is None or u.team == team)]

    def unit_by_id(self, entity_id: int) -> Unit | None:
        for unit in self.units:
            if unit.entity_id == entity_id:
                return unit
        return None

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    def advance(self, ticks: int | None = None) -> None:
        steps = self.config.decision_interval if ticks is None else ticks
        for _ in range(steps):
            if self.done:
                break
            self._tick()
        self._pending_reward = self._reward_after_window()

    def step_decision(
        self,
        actions: Sequence[dict[str, Any]],
        team: int = 0,
    ) -> dict[str, Any]:
        """Apply one decision, run the opponent, advance a decision window."""
        self.events = []
        receipts = self.apply_actions(actions, team=team)
        opponent_actions = self._opponent_actions(team=team)
        if opponent_actions:
            self.apply_actions(opponent_actions, team=1)
        self.advance()
        state = self.observation(team=team)
        state.update(
            {
                "reward": self._pending_reward,
                "reward_components": dict(getattr(self, "latest_reward_components", {})),
                "terminated": self.done and self.result in {"victory", "defeat"},
                "truncated": self.done and self.result == "timeout",
                "result": self.result,
                "receipts": receipts,
                "events": list(self.events),
                "decision_id": None,
            }
        )
        return state

    def _tick(self) -> None:
        cfg = self.config
        self.tick += 1
        if cfg.enable_economy:
            self._economic_tick()

        for unit in self.alive_units():
            if unit.cooldown > 0:
                unit.cooldown -= 1
            target = self._select_target(unit)
            if (
                target is not None
                and unit.cooldown == 0
                and self._distance(unit, target) <= cfg.attack_range
            ):
                unit.cooldown = cfg.attack_cooldown_ticks
                self._damage(target, cfg.unit_damage, source=unit)
            self._move(unit)
        self._check_done()

    # ------------------------------------------------------------------
    # Rewards and diagnostics
    # ------------------------------------------------------------------
    def snapshot(self) -> tuple[Any, ...]:
        """Deterministic fingerprint used by replay/regression tests."""
        return (
            self.tick,
            self.result,
            tuple(
                (
                    u.entity_id,
                    u.team,
                    round(u.x, 3),
                    round(u.y, 3),
                    round(u.hp, 3),
                    u.order,
                    u.target_id,
                    u.alive,
                )
                for u in self.units
            ),
        )

    def evaluate(self, team: int = 0) -> dict[str, Any]:
        cfg = self.config
        total = float(cfg.units_per_team * cfg.unit_hp)
        return {
            "tick": self.tick,
            "result": self.result,
            "own_hp_frac": sum(u.hp for u in self.alive_units(team)) / total,
            "enemy_hp_frac": sum(u.hp for u in self.alive_units(1 - team)) / total,
            "own_alive": len(self.alive_units(team)),
            "enemy_alive": len(self.alive_units(1 - team)),
            "damage_dealt": self.stats["damage_dealt"][team],
            "damage_taken": self.stats["damage_taken"][team],
            "kills": self.stats["kills"][team],
            "losses": self.stats["losses"][team],
        }
