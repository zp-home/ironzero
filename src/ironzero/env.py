# SPDX-License-Identifier: Apache-2.0
"""Gymnasium-style environment for the micro-duel reference scenario.

The environment owns the boundary between "Python decision space" and
"game-side adapter". It converts a discrete ``MultiDiscrete`` action tuple
into ``ironzero/1`` protocol actions, applies action masks, and returns flat
``float32`` observations together with reward / terminated / truncated.

It intentionally avoids a hard dependency on ``gymnasium`` so the project can
run in an offline environment; the API mirrors the Gymnasium contract and
``register()`` is attempted when the package is installed.
"""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np

from .micro_config import DEFAULT_CONFIG, MicroDuelConfig
from .mock_engine import ACTION_KINDS, EXTRACTOR_TYPE_ID, MicroDuelAdapter
from .observations import build_masks, encode_observation


class BoxSpace:
    """Minimal Box-compatible observation space."""

    def __init__(self, shape: tuple[int, ...], dtype: Any = np.float32) -> None:
        self.shape = shape
        self.dtype = dtype
        self.low = np.full(shape, -np.inf, dtype=dtype)
        self.high = np.full(shape, np.inf, dtype=dtype)

    def sample(self) -> np.ndarray:  # pragma: no cover - used by diagnostics only
        return np.zeros(self.shape, dtype=self.dtype)

    def contains(self, value: np.ndarray) -> bool:
        return value.shape == self.shape


class MultiDiscreteSpace:
    """Minimal MultiDiscrete-compatible action space."""

    def __init__(self, nvec: tuple[int, ...]) -> None:
        self.nvec = np.asarray(nvec, dtype=np.int64)

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self.nvec),)

    def sample(self) -> np.ndarray:
        return np.array([np.random.randint(0, int(n)) for n in self.nvec], dtype=np.int64)

    def contains(self, value: np.ndarray) -> bool:
        return (
            isinstance(value, np.ndarray)
            and value.shape == self.shape
            and np.all(value >= 0)
            and np.all(value < self.nvec)
        )


class MicroDuelEnv:
    """Single-agent micro-duel environment (team 0 vs scripted team 1)."""

    metadata = {"render_modes": [], "name": "IronZeroMicroDuel-v0"}

    def __init__(
        self,
        config: MicroDuelConfig | None = None,
        opponent: str | None = None,
        adapter: MicroDuelAdapter | None = None,
        seed: int | None = None,
        episode_prefix: str = "micro",
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        self.adapter = adapter or MicroDuelAdapter(
            config=self.config, opponent=opponent or self.config.opponent
        )
        self._episode_counter = 0
        self._episode_seed = int(seed or 0)
        self.episode_prefix = episode_prefix
        self.state: dict[str, Any] = {}
        self.masks: dict[str, np.ndarray] = {}
        self.episode_id = ""
        self.observation_space = BoxSpace((self.config.flat_obs_dim,))
        self.action_space = MultiDiscreteSpace(self.config.action_nvec)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._episode_counter += 1
        if seed is not None:
            self._episode_seed = int(seed)
        elif self._episode_counter > 1:
            self._episode_seed += 1
        episode_id = options.get("episode_id") if options else None
        self.episode_id = episode_id or f"{self.episode_prefix}-{self._episode_seed}-{self._episode_counter}"
        self.state = self.adapter.reset(
            episode_id=self.episode_id,
            seed=self._episode_seed,
            scenario="micro_duel",
        )
        self.masks = build_masks(self.state, self.config)
        obs = encode_observation(self.state, self.config)
        info = {
            "episode_id": self.episode_id,
            "seed": self._episode_seed,
            "masks": self.masks,
            "raw_state": self.state,
        }
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.int64).reshape(-1)
        if action.shape != self.action_space.shape:
            raise ValueError(f"action must have shape {self.action_space.shape}, got {action.shape}")
        return self.step_actions(self.decode_actions(action))

    def step_actions(self, protocol_actions: list[dict]):
        """Step with already-decoded protocol actions (scripted/demo policies)."""

        result = self.adapter.step(protocol_actions)
        self.state = result
        self.masks = build_masks(self.state, self.config)
        obs = encode_observation(self.state, self.config)
        reward = float(result.get("reward", 0.0))
        terminated = bool(result.get("terminated", False))
        truncated = bool(result.get("truncated", False))
        info = {
            "episode_id": self.episode_id,
            "tick": result.get("tick", 0),
            "result": result.get("result"),
            "receipts": result.get("receipts", []),
            "masks": self.masks,
            "raw_state": self.state,
            "eval": self._adapter_eval(),
            "reward_components": result.get("reward_components") or {},
        }
        return obs, reward, terminated, truncated, info

    def _adapter_eval(self) -> dict | None:
        evaluate = getattr(self.adapter, "evaluate", None)
        if evaluate is None:
            return None
        try:
            return evaluate(0)
        except TypeError:
            return evaluate()

    def action_masks(self) -> dict[str, np.ndarray]:
        return self.masks

    def close(self) -> None:
        self.adapter.close()

    # ------------------------------------------------------------------
    # Action decoding
    # ------------------------------------------------------------------
    def decode_action(self, action: np.ndarray) -> dict[str, Any]:
        """Decode one decision into a single protocol action (first group)."""

        return self.decode_actions(action)[0]

    def decode_actions(self, action: np.ndarray) -> list[dict[str, Any]]:
        """Decode one decision into one or more protocol actions.

        The last unit-group slot means "all alive groups" (broadcast): that is
        the macro action the policy normally uses to move or focus-fire the
        whole army with a single decision.
        """

        cfg = self.config
        kind_index = int(action[0]) % len(ACTION_KINDS)
        group_index, gx, gy, target_index = (int(v) for v in action[1:5])
        kind = ACTION_KINDS[kind_index]
        groups = self._target_groups(group_index)
        state = self.state
        enemies = state.get("visible_enemy_entities", [])
        target_ids = state.get("target_ids", [])

        if kind in {"move_group", "attack_move"}:
            gx = int(np.clip(gx, 0, cfg.position_grid - 1))
            gy = int(np.clip(gy, 0, cfg.position_grid - 1))
            x, y = cfg.grid_to_world(gx, gy)
            return [{"kind": kind, "unit_group": group, "x": x, "y": y} for group in groups]

        if kind == "attack_target":
            if target_index < len(target_ids) and target_index < len(enemies):
                target_id = int(target_ids[target_index])
                return [
                    {"kind": kind, "unit_group": group, "target_entity_id": target_id}
                    for group in groups
                ]
            # "no target" slot or stale index -> safe fallback
            return [{"kind": "stop", "unit_group": group} for group in groups]

        if kind == "build_extractor":
            # The target head doubles as the resource-slot selector. Slots are
            # a single global list, so this is a single action (not per group).
            slots = state.get("resource_slots") or []
            if not slots:
                return [{"kind": "stop", "unit_group": groups[0]}]
            slot_id = int(target_index) % len(slots)
            return [{"kind": "build_extractor", "slot_id": slot_id}]

        if kind == "build_structure":
            gx = int(np.clip(gx, 0, cfg.position_grid - 1))
            gy = int(np.clip(gy, 0, cfg.position_grid - 1))
            x, y = cfg.grid_to_world(gx, gy)
            return [
                {
                    "kind": "build_structure",
                    "structure_type": "land_factory",
                    "x": x,
                    "y": y,
                }
            ]

        if kind == "produce_unit":
            factories = [
                u for u in state.get("own_entities", [])
                if u.get("is_factory") or u.get("is_factory_building")
            ]
            if not factories:
                return [{"kind": "stop", "unit_group": groups[0]}]
            factory = factories[int(target_index) % len(factories)]
            return [
                {
                    "kind": "produce_unit",
                    "factory_entity_id": int(factory.get("entity_id", -1)),
                    "unit_type": "tank",
                }
            ]

        if kind == "upgrade_building":
            extractors = [
                u for u in state.get("own_entities", [])
                if (u.get("is_building") and int(u.get("entity_type_id") or 0) == EXTRACTOR_TYPE_ID)
            ]
            if not extractors:
                return [{"kind": "stop", "unit_group": groups[0]}]
            building = extractors[int(target_index) % len(extractors)]
            return [
                {
                    "kind": "upgrade_building",
                    "building_entity_id": int(building.get("entity_id", -1)),
                }
            ]

        if kind == "repair_target":
            if target_index < len(target_ids) and target_index < len(enemies):
                return [
                    {"kind": "repair_target", "target_entity_id": int(target_ids[target_index])}
                ]
            return [{"kind": "stop", "unit_group": groups[0]}]

        return [{"kind": kind, "unit_group": group} for group in groups]

    def _target_groups(self, group_index: int) -> list[int]:
        cfg = self.config
        broadcast = cfg.groups  # last slot = all groups
        if group_index >= broadcast:
            return list(range(cfg.groups))
        return [self._safe_group(group_index)]

    def _safe_group(self, group_index: int) -> int:
        cfg = self.config
        if self.masks and self.masks["group"][group_index % cfg.groups]:
            return group_index % cfg.groups
        alive = [
            int(index)
            for index, valid in enumerate(self.masks["group"][: cfg.groups]) if valid
        ] if self.masks else []
        return alive[0] if alive else 0


def make_env(
    config: MicroDuelConfig | None = None,
    opponent: str | None = None,
    seed: int = 0,
    **kwargs: Any,
) -> MicroDuelEnv:
    return MicroDuelEnv(config=config, opponent=opponent, seed=seed, **kwargs)


def _register_gymnasium() -> None:
    try:  # pragma: no cover - optional integration
        import gymnasium

        gymnasium.register(id="IronZeroMicroDuel-v0", entry_point="ironzero.env:MicroDuelEnv")
    except Exception:  # pragma: no cover - offline fallback
        return


_register_gymnasium()

__all__ = [
    "BoxSpace",
    "MultiDiscreteSpace",
    "MicroDuelEnv",
    "make_env",
    "uuid",
]
