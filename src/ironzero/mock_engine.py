# SPDX-License-Identifier: Apache-2.0
"""Deterministic micro-duel engine used as the reference game adapter.

This module plays the role of the in-game side of the bridge: it owns the
simulation, validates and executes actions, advances ticks and produces raw
observation payloads that are shaped like ``ironzero/1`` messages.

It is intentionally dependency-free (standard library only) and fully
deterministic given ``(scenario, seed)``. That determinism is what lets the
project test the *whole* pipeline before the real RWPP headless runtime is
available, and it doubles as the fixed-baseline environment for CI.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .micro_config import DEFAULT_CONFIG, MicroDuelConfig

ORDER_NAMES: dict[int, str] = {
    0: "idle",
    1: "move",
    2: "attack_move",
    3: "attack_target",
    4: "hold",
}

ORDER_IDS: dict[str, int] = {name: order for order, name in ORDER_NAMES.items()}

ACTION_KINDS: tuple[str, ...] = (
    "move_group",
    "attack_move",
    "attack_target",
    "stop",
    "hold",
)

TANK_TYPE_ID = 1

# Rejection reasons mirror ironzero.protocol.ErrorCode on purpose.
STALE_TICK = "STALE_TICK"
UNKNOWN_UNIT = "UNKNOWN_UNIT"
UNIT_NOT_OWNED = "UNIT_NOT_OWNED"
TARGET_NOT_VISIBLE = "TARGET_NOT_VISIBLE"
POSITION_BLOCKED = "POSITION_BLOCKED"
ACTION_MASKED = "ACTION_MASKED"
INVALID_MESSAGE = "INVALID_MESSAGE"


@dataclass(slots=True)
class Unit:
    entity_id: int
    team: int
    group: int
    x: float
    y: float
    max_hp: float
    hp: float
    order: int = 0
    tx: float = 0.0
    ty: float = 0.0
    target_id: int | None = None
    cooldown: int = 0
    alive: bool = True


class MicroDuelEngine:
    """Two teams of tanks on an open map, micro control only."""

    def __init__(
        self,
        config: MicroDuelConfig | None = None,
        opponent: str | None = None,
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        self.opponent = opponent or self.config.opponent
        if self.opponent not in {"scripted", "idle", "random"}:
            raise ValueError(f"unknown opponent: {self.opponent}")
        self._rng = random.Random(0)
        self.units: list[Unit] = []
        self.tick = 0
        self.done = False
        self.result: str | None = None
        self.stats: dict[str, Any] = {}
        self._phi_prev = 0.0
        self._pending_reward = 0.0
        self.reset(seed=0)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self, seed: int) -> None:
        cfg = self.config
        self._rng = random.Random(seed)
        self.tick = 0
        self.done = False
        self.result = None
        self.units = []
        self.stats = {
            "damage_dealt": {team: 0.0 for team in range(cfg.teams)},
            "damage_taken": {team: 0.0 for team in range(cfg.teams)},
            "kills": {team: 0 for team in range(cfg.teams)},
            "losses": {team: 0 for team in range(cfg.teams)},
        }
        per_team = cfg.units_per_team
        rows = max(1, math.ceil(per_team / 2))
        spacing_y = min(72.0, (cfg.map_height - 2 * cfg.spawn_margin) / max(1, rows - 1)) if rows > 1 else 0.0
        entity_id = 1
        for team in range(cfg.teams):
            side_x = cfg.spawn_margin if team == 0 else cfg.map_width - cfg.spawn_margin
            for index in range(per_team):
                column = index % 2
                row = index // 2
                jitter = self._rng.uniform(-10.0, 10.0)
                x = side_x + (18.0 if team == 0 else -18.0) * column + jitter
                y = cfg.map_height / 2.0 + (row - (rows - 1) / 2.0) * spacing_y + jitter
                group = min(index // max(1, per_team // cfg.groups), cfg.groups - 1)
                self.units.append(
                    Unit(
                        entity_id=entity_id,
                        team=team,
                        group=group,
                        x=x,
                        y=y,
                        max_hp=cfg.unit_hp,
                        hp=cfg.unit_hp,
                        tx=x,
                        ty=y,
                    )
                )
                entity_id += 1
        self._phi_prev = self._potential()
        self._pending_reward = 0.0

    def alive_units(self, team: int | None = None) -> list[Unit]:
        return [u for u in self.units if u.alive and (team is None or u.team == team)]

    def unit_by_id(self, entity_id: int) -> Unit | None:
        for unit in self.units:
            if unit.entity_id == entity_id:
                return unit
        return None

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def apply_actions(
        self,
        actions: Sequence[dict[str, Any]],
        team: int = 0,
    ) -> list[dict[str, Any]]:
        receipts: list[dict[str, Any]] = []
        if self.done:
            return [
                {"accepted": False, "rejected_reason": STALE_TICK, "action_index": i}
                for i, _ in enumerate(actions)
            ]
        for index, action in enumerate(actions[:64]):
            reason = self._apply_one(action, team)
            receipts.append(
                {
                    "action_index": index,
                    "kind": action.get("kind"),
                    "accepted": reason is None,
                    "rejected_reason": reason,
                }
            )
        return receipts

    def _apply_one(self, action: dict[str, Any], team: int) -> str | None:
        kind = action.get("kind")
        if kind not in ACTION_KINDS:
            return INVALID_MESSAGE
        group = action.get("unit_group")
        if not isinstance(group, int) or not 0 <= group < self.config.groups:
            return UNKNOWN_UNIT
        members = [u for u in self.alive_units(team) if u.group == group]
        if not members:
            return UNIT_NOT_OWNED

        if kind in {"move_group", "attack_move"}:
            x = float(action.get("x", math.nan))
            y = float(action.get("y", math.nan))
            if not (math.isfinite(x) and math.isfinite(y)):
                return INVALID_MESSAGE
            if not (0.0 <= x <= self.config.map_width and 0.0 <= y <= self.config.map_height):
                return POSITION_BLOCKED
            order = 1 if kind == "move_group" else 2
            for member in members:
                member.order = order
                member.tx = x
                member.ty = y
                member.target_id = None
            return None

        if kind == "attack_target":
            target_id = action.get("target_entity_id")
            if not isinstance(target_id, int) or target_id < 0:
                return INVALID_MESSAGE
            target = self.unit_by_id(target_id)
            if target is None or not target.alive or target.team == team:
                return TARGET_NOT_VISIBLE
            for member in members:
                member.order = 3
                member.target_id = target_id
            return None

        if kind == "stop":
            for member in members:
                member.order = 0
                member.target_id = None
                member.tx = member.x
                member.ty = member.y
            return None

        # hold
        for member in members:
            member.order = 4
            member.target_id = None
            member.tx = member.x
            member.ty = member.y
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
        receipts = self.apply_actions(actions, team=team)
        opponent_actions = self._opponent_actions(team=team)
        if opponent_actions:
            self.apply_actions(opponent_actions, team=1)
        self.advance()
        state = self.observation(team=team)
        state.update(
            {
                "reward": self._pending_reward,
                "terminated": self.done and self.result in {"victory", "defeat"},
                "truncated": self.done and self.result == "timeout",
                "result": self.result,
                "receipts": receipts,
                "decision_id": None,
            }
        )
        return state

    def _tick(self) -> None:
        cfg = self.config
        self.tick += 1
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

    def _select_target(self, unit: Unit) -> Unit | None:
        if unit.order == 1:  # plain move never auto-engages
            return None
        if unit.order == 3 and unit.target_id is not None:
            target = self.unit_by_id(unit.target_id)
            if target is not None and target.alive:
                return target
            unit.order = 0
            unit.target_id = None
        best: Unit | None = None
        best_dist = math.inf
        for enemy in self.alive_units(1 - unit.team):
            dist = self._distance(unit, enemy)
            if dist < best_dist:
                best, best_dist = enemy, dist
        return best

    def _move(self, unit: Unit) -> None:
        cfg = self.config
        speed = cfg.unit_speed / cfg.tick_rate
        if unit.order == 0:
            return
        if unit.order == 4:
            return
        target = None
        if unit.order == 3 and unit.target_id is not None:
            target = self.unit_by_id(unit.target_id)
            if target is None or not target.alive:
                unit.order = 0
                return
        goal_x = target.x if target is not None else unit.tx
        goal_y = target.y if target is not None else unit.ty
        dx, dy = goal_x - unit.x, goal_y - unit.y
        distance = math.hypot(dx, dy)
        if unit.order == 3 and target is not None:
            stop_at = cfg.attack_range * 0.85
        else:
            stop_at = 2.0
        if distance <= stop_at:
            if unit.order == 2 and target is None:
                unit.order = 0
            return
        step = min(speed, distance - stop_at + 1e-9)
        unit.x += dx / distance * step
        unit.y += dy / distance * step
        unit.x = _clamp(unit.x, 0.0, cfg.map_width)
        unit.y = _clamp(unit.y, 0.0, cfg.map_height)

    @staticmethod
    def _distance(a: Unit, b: Unit) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)

    def _damage(self, target: Unit, amount: float, source: Unit) -> None:
        if not target.alive:
            return
        actual = min(target.hp, amount)
        target.hp -= actual
        self.stats["damage_dealt"][source.team] += actual
        self.stats["damage_taken"][target.team] += actual
        if target.hp <= 0.0:
            target.alive = False
            target.hp = 0.0
            self.stats["kills"][source.team] += 1
            self.stats["losses"][target.team] += 1

    def _check_done(self) -> None:
        cfg = self.config
        if self.done:
            return
        if not self.alive_units(1):
            self.done = True
            self.result = "victory"
        elif not self.alive_units(0):
            self.done = True
            self.result = "defeat"
        elif self.tick >= cfg.episode_ticks:
            self.done = True
            self.result = "timeout"

    # ------------------------------------------------------------------
    # Scripted opponents
    # ------------------------------------------------------------------
    def _opponent_actions(self, team: int = 0) -> list[dict[str, Any]]:
        if self.done:
            return []
        enemy_team = 1 - team
        own = self.alive_units(enemy_team)
        if not own:
            return []
        targets = self.alive_units(team)
        if not targets:
            return []
        if self.opponent == "idle":
            return []
        if self.opponent == "random":
            actions: list[dict[str, Any]] = []
            for group in sorted({u.group for u in own}):
                member = next(u for u in own if u.group == group)
                kind = self._rng.choice(ACTION_KINDS)
                if kind in {"move_group", "attack_move"}:
                    actions.append(
                        {
                            "kind": kind,
                            "unit_group": group,
                            "x": self._rng.uniform(0.0, self.config.map_width),
                            "y": self._rng.uniform(0.0, self.config.map_height),
                        }
                    )
                elif kind == "attack_target":
                    actions.append(
                        {
                            "kind": kind,
                            "unit_group": group,
                            "target_entity_id": self._rng.choice(targets).entity_id,
                        }
                    )
                else:
                    actions.append({"kind": kind, "unit_group": group})
                _ = member
            return actions
        # scripted aggressor: every group focuses the nearest enemy
        actions = []
        for group in sorted({u.group for u in own}):
            members = [u for u in own if u.group == group]
            anchor = members[0]
            nearest = min(targets, key=lambda t: self._distance(anchor, t))
            actions.append(
                {
                    "kind": "attack_target",
                    "unit_group": group,
                    "target_entity_id": nearest.entity_id,
                }
            )
        return actions

    # ------------------------------------------------------------------
    # Observation payload (shape-compatible with ironzero/1 messages)
    # ------------------------------------------------------------------
    def observation(self, team: int = 0) -> dict[str, Any]:
        cfg = self.config
        own = self.alive_units(team)
        enemies = self.alive_units(1 - team)
        own_hp = sum(u.hp for u in own)
        enemy_hp = sum(u.hp for u in enemies)
        return {
            "tick": self.tick,
            "game_time": self.tick / cfg.tick_rate,
            "controlled_team": team,
            "global_state": {
                "unit_count": len(own),
                "building_count": 0,
                "army_value": round(own_hp, 3),
                "visible_enemy_value": round(enemy_hp, 3),
                "kills": self.stats["kills"][team],
                "losses": self.stats["losses"][team],
                "income": 0.0,
                "funds": 0.0,
            },
            "own_entities": [self._entity_payload(u) for u in own],
            "visible_enemy_entities": [self._entity_payload(u) for u in enemies],
            "remembered_enemies": [],
            "local_maps": {},
            "target_ids": [u.entity_id for u in enemies],
            "tick_interval": cfg.decision_interval,
        }

    def _entity_payload(self, unit: Unit) -> dict[str, Any]:
        return {
            "entity_id": unit.entity_id,
            "entity_type_id": TANK_TYPE_ID,
            "team_id": unit.team,
            "x": round(unit.x, 4),
            "y": round(unit.y, 4),
            "hp": round(unit.hp, 4),
            "max_hp": unit.max_hp,
            "is_building": False,
            "is_visible": True,
            "is_alive": unit.alive,
            "order_type": ORDER_NAMES[unit.order],
            "attack_target_id": unit.target_id,
            "group_id": unit.group,
            "cooldown_frac": round(unit.cooldown / self.config.attack_cooldown_ticks, 4),
        }

    # ------------------------------------------------------------------
    # Rewards and diagnostics
    # ------------------------------------------------------------------
    def _potential(self) -> float:
        cfg = self.config
        total = float(cfg.units_per_team * cfg.unit_hp)
        own = sum(u.hp for u in self.alive_units(0)) / total
        enemy = sum(u.hp for u in self.alive_units(1)) / total
        return own - enemy

    def _reward_after_window(self) -> float:
        cfg = self.config
        phi = self._potential()
        reward = cfg.potential_coef * (phi - self._phi_prev)
        self._phi_prev = phi
        if self.done:
            if self.result == "victory":
                reward += cfg.win_reward
            elif self.result == "defeat":
                reward += cfg.loss_reward
        return reward

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


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


class MicroDuelAdapter:
    """Game-side adapter interface shared by the env and the TCP bridge."""

    def __init__(self, config: MicroDuelConfig | None = None, opponent: str | None = None) -> None:
        self.engine = MicroDuelEngine(config=config, opponent=opponent)
        self.opponent = self.engine.opponent
        self.episode_id: str = "ep-0"

    def reset(self, *, episode_id: str, seed: int, scenario: str = "micro_duel") -> dict[str, Any]:
        if scenario != "micro_duel":
            raise ValueError(f"unsupported scenario: {scenario}")
        self.episode_id = episode_id
        self.engine.reset(seed=seed)
        return self.engine.observation()

    def step(self, actions: Iterable[dict[str, Any]], *, team: int = 0) -> dict[str, Any]:
        return self.engine.step_decision(list(actions), team=team)

    def evaluate(self, team: int = 0) -> dict[str, Any]:
        return self.engine.evaluate(team)

    def close(self) -> None:  # pragma: no cover - symmetry with TCP adapters
        return None
