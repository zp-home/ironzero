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
from ..mock_engine import ACTION_KINDS


def observation_to_state(message: dict[str, Any]) -> dict[str, Any]:
    """Convert one bridge ``obs`` message into the encoder's raw state dict.

    The real bridge only reports positions/hp for alive units; group and order
    metadata are optional and default to group 0 / idle in the encoder.

    ``resource_slots`` must be carried through: both the encoder (how many
    extractors are placed) and :func:`decode_action` (which pool a
    ``build_extractor`` order targets) read it. Dropping it here made every
    extractor order resolve to slot 0 and told the policy that no extractor was
    ever built, so builders were free to wander off to the enemy base instead of
    expanding the economy.
    """

    enemies = [dict(entity) for entity in message.get("visible_enemy_entities", [])]
    info = message.get("info")
    slots = message.get("resource_slots")
    state = {
        "tick": int(message.get("tick", 0)),
        "game_time": float(message.get("game_time", 0.0)),
        "controlled_team": int(message.get("controlled_team", 0)),
        "global_state": dict(message.get("global_state") or {}),
        "own_entities": [dict(entity) for entity in message.get("own_entities", [])],
        "visible_enemy_entities": enemies,
        "target_ids": [int(entity.get("entity_id", -1)) for entity in enemies],
        # Resource pools: ``slot_id``/``x``/``y``/``occupied``/``extractor_level``.
        # ``build_extractor`` resolves ``slot_id`` against this list, and the
        # encoder counts occupied slots from it. Absent (older bridge) -> empty.
        "resource_slots": [dict(slot) for slot in slots] if isinstance(slots, list) else [],
        # ``info.builders`` is the engine's own answer to "which economy unit
        # types can our builders queue". The action mask reads it so a kind is
        # only selectable when the game would actually accept it; without this
        # the policy keeps picking extractor orders that the map cannot honour.
        "builders": list((info or {}).get("builders") or []) if isinstance(info, dict) else [],
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
    # Enemies keep the bridge's stable stream order: the encoder writes them in
    # that order, so target slot i names the i-th streamed enemy and the mapping
    # does not shuffle between frames.
    enemies = state.get("visible_enemy_entities", [])
    own = state.get("own_entities", [])

    if kind in {"move_group", "attack_move"}:
        x, y = config.grid_to_world(gx, gy)
        return [{"kind": kind, "unit_group": group, "x": round(float(x), 2), "y": round(float(y), 2)}]
    if kind == "build_extractor":
        # ``slot_id`` indexes the resource slots reported by the bridge. Only
        # slots without an extractor are candidates: re-ordering an existing pool
        # is a no-op the engine accepts, which would let the policy "succeed"
        # forever without ever growing the economy. Among the free slots the
        # bridge already sorts nearest-to-own-base first, so the counter keeps
        # that preference. When the bridge reports no slots at all we still emit
        # the action and let the Java side resolve slot 0 against its own map
        # resource list.
        slots = state.get("resource_slots") or []
        free = [
            index
            for index, slot in enumerate(slots)
            if not (isinstance(slot, dict) and slot.get("occupied"))
        ]
        if free:
            # The target head is shared with the enemy head, so a slot index is
            # taken modulo the number of free pools rather than trusted raw.
            candidates = free
        elif slots:
            # Every pool is taken: fall back to all of them so the order is still
            # well-formed (the engine rejects it as already occupied).
            candidates = list(range(len(slots)))
        else:
            candidates = [0]
        return [{"kind": "build_extractor", "unit_group": group,
                 "slot_id": int(candidates[int(target_index) % len(candidates)])}]
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
        # Select the target by what the bridge says the engine would accept, not
        # by a hard-coded type id. The old test was
        # ``entity_type_id == EXTRACTOR_TYPE_ID (10)``, but the bridge used to
        # report every building as ``entity_type_id = 2``, so this branch never
        # found a building and degraded every upgrade order into ``hold`` -- the
        # extractor tier upgrade, the core of the Rusted Warfare economy, simply
        # did not exist on the real game.
        #
        # ``can_upgrade`` is the engine's own answer ("this building has a next
        # tier available right now", from ``getBuildActionId``). Bridges that do
        # not report it fall back to any own building, which at worst yields a
        # rejected order instead of a silently dropped one.
        upgradable = [u for u in own if u.get("can_upgrade")]
        if not upgradable:
            upgradable = [u for u in own if u.get("is_building")]
        if not upgradable:
            return _hold_action(state, group)
        building = upgradable[int(target_index) % len(upgradable)]
        return [{"kind": "upgrade_building", "unit_group": group,
                 "building_entity_id": int(building.get("entity_id", -1))}]
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
    if kind == "noop":
        # Issue nothing at all. Construction and production need hundreds of
        # uninterrupted ticks, and every order cancels the in-flight one, so the
        # policy must be able to skip a decision without stopping its builders.
        return []
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
    # Resource pools belong to the terminal state too; the encoder counts them.
    if isinstance(message.get("resource_slots"), list):
        merged["resource_slots"] = [dict(slot) for slot in message["resource_slots"]]
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
