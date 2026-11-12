# SPDX-License-Identifier: Apache-2.0
"""Bridge observation -> encoder state, and policy action -> ``ironzero/1`` steps.

Pure conversion helpers: no socket, no process, no reward bookkeeping. Keeping
them side-effect free is what lets the tests exercise the protocol contract
without a game runtime.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..micro_config import MicroDuelConfig
from ..mock_engine import ACTION_KINDS, EXTRACTOR_TYPE_ID


def observation_to_state(message: dict[str, Any]) -> dict[str, Any]:
    """Convert one bridge ``obs`` message into the encoder's raw state dict.

    The real bridge only reports positions/hp for alive units; group and order
    metadata are optional and default to group 0 / idle in the encoder.
    """

    enemies = [dict(entity) for entity in message.get("visible_enemy_entities", [])]
    state = {
        "tick": int(message.get("tick", 0)),
        "game_time": float(message.get("game_time", 0.0)),
        "controlled_team": int(message.get("controlled_team", 0)),
        "global_state": dict(message.get("global_state") or {}),
        "own_entities": [dict(entity) for entity in message.get("own_entities", [])],
        "visible_enemy_entities": enemies,
        "target_ids": [int(entity.get("entity_id", -1)) for entity in enemies],
        "receipts": [],
        "result": message.get("result"),
        "terminated": bool(message.get("terminated", False)),
        "truncated": bool(message.get("truncated", False)),
    }
    return state


def decode_action(
    action: np.ndarray,
    state: dict[str, Any],
    config: MicroDuelConfig,
) -> list[dict[str, Any]]:
    """Decode one policy action into ``ironzero/1`` step actions.

    The bridge understands the same group head as the mock engine: 0..G-1 order
    the units whose id maps to that group, and the last slot (G) broadcasts to
    every living friendly unit.
    """

    vector = np.asarray(action, dtype=np.int64).reshape(-1)
    expected_heads = len(config.action_nvec)
    if vector.shape[0] != expected_heads:
        raise ValueError(
            f"action must have {expected_heads} heads, got {vector.shape[0]}"
        )
    kind = ACTION_KINDS[int(vector[0]) % len(ACTION_KINDS)]
    group = int(vector[1])
    gx = int(np.clip(vector[2], 0, config.position_grid - 1))
    gy = int(np.clip(vector[3], 0, config.position_grid - 1))
    target_index = int(vector[4])
    enemies = state.get("visible_enemy_entities", [])
    own = state.get("own_entities", [])

    if kind in {"move_group", "attack_move"}:
        x, y = config.grid_to_world(gx, gy)
        return [{"kind": kind, "unit_group": group, "x": round(float(x), 2), "y": round(float(y), 2)}]
    if kind == "build_extractor":
        # ``slot_id`` indexes the resource slots reported by the bridge. When the
        # bridge reports none we still emit the action and let the game decide:
        # the Java side resolves slot 0 against its own map resource list.
        slots = state.get("resource_slots") or [{}]
        return [{"kind": "build_extractor", "unit_group": group,
                 "slot_id": int(target_index) % len(slots)}]
    if kind == "build_structure":
        x, y = config.grid_to_world(gx, gy)
        return [{"kind": "build_structure", "unit_group": group,
                 "structure_type": "land_factory",
                 "x": round(float(x), 2), "y": round(float(y), 2)}]
    if kind == "produce_unit":
        factories = [u for u in own if u.get("is_factory") or u.get("is_factory_building")]
        if not factories:
            return _hold_action(state, group)
        factory = factories[int(target_index) % len(factories)]
        return [{"kind": "produce_unit", "unit_group": group,
                 "factory_entity_id": int(factory.get("entity_id", -1)),
                 "unit_type": "tank"}]
    if kind == "upgrade_building":
        buildings = [
            u for u in own
            if u.get("is_building") and int(u.get("entity_type_id") or 0) == EXTRACTOR_TYPE_ID
        ]
        if not buildings:
            return _hold_action(state, group)
        building = buildings[int(target_index) % len(buildings)]
        return [{"kind": "upgrade_building", "unit_group": group,
                 "building_entity_id": int(building.get("entity_id", -1))}]
    if kind == "repair_target":
        if 0 <= target_index < len(enemies):
            return [{"kind": "repair_target", "unit_group": group,
                     "target_entity_id": int(enemies[target_index].get("entity_id", -1))}]
        return _hold_action(state, group)
    if kind == "attack_target":
        if 0 <= target_index < len(enemies):
            return [
                {
                    "kind": kind,
                    "unit_group": group,
                    "target_entity_id": int(enemies[target_index].get("entity_id", -1)),
                }
            ]
        return _hold_action(state, group)
    if kind in {"stop", "hold"}:
        return _hold_action(state, group)
    return [{"kind": kind, "unit_group": group}]


def _hold_action(state: dict[str, Any], group: int = 4) -> list[dict[str, Any]]:
    """Stop/hold order; the bridge keeps each unit at its current position."""

    return [{"kind": "hold", "unit_group": int(group)}]


def merge_done_message(state: dict[str, Any], message: dict[str, Any]) -> dict[str, Any]:
    """Fold a ``done`` message into the previous state (terminal-observation path).

    A ``done`` message ends the episode. Some bridge builds attach the final
    entities/tick to it, others only the result: merge whatever is present so the
    returned observation/masks describe the real terminal state instead of the
    previous decision's state.
    """

    merged = dict(state)
    if message.get("tick") is not None:
        try:
            merged["tick"] = int(message["tick"])
        except (TypeError, ValueError):
            pass
    if message.get("game_time") is not None:
        merged["game_time"] = float(message["game_time"])
    if message.get("result") is not None:
        merged["result"] = message["result"]
    if message.get("terminated") is not None:
        merged["terminated"] = bool(message["terminated"])
    if message.get("truncated") is not None:
        merged["truncated"] = bool(message["truncated"])
    for key in ("own_entities", "visible_enemy_entities"):
        if isinstance(message.get(key), list):
            merged[key] = [dict(entity) for entity in message[key]]
    if isinstance(message.get("own_entities"), list) or isinstance(
        message.get("visible_enemy_entities"), list
    ):
        merged["target_ids"] = [
            int(entity.get("entity_id", -1))
            for entity in merged.get("visible_enemy_entities", [])
        ]
    return merged


__all__ = [
    "_hold_action",
    "decode_action",
    "merge_done_message",
    "observation_to_state",
]
