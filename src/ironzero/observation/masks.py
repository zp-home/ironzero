# SPDX-License-Identifier: Apache-2.0
"""Action masks for the micro-duel policy heads.

The mask layout mirrors ``MicroDuelConfig.action_nvec``:
kind(K), group(G), x(P), y(P), target(T), where ``K == len(ACTION_KINDS)``.
Masks are derived from the same payload as the observation vector so the two
stay aligned; nothing here is learned or normalised.
"""

from __future__ import annotations

import numpy as np

from ..micro_config import DEFAULT_CONFIG, MicroDuelConfig
from ..mock_engine import ACTION_KINDS, BUILDER_TYPE_ID

__all__ = ["build_masks", "flat_mask_vector", "masks_as_lists"]

# Which ``info.builders[].can_build`` entry each economy kind needs. The real
# game decides what a builder may queue; the mask has to agree with it, or the
# policy learns "these kinds are selectable" and then pays ``invalid_penalty``
# on every attempt the engine rejects. On stock skirmish maps the built-in
# ``builder`` reports only ``landFactory``, so extractor/upgrade/tank stay
# masked until the map actually offers them.
_KIND_REQUIRED_BUILDABLE: dict[str, str] = {
    "build_extractor": "extractor",
    "build_structure": "landFactory",
    "produce_unit": "tank",
    "upgrade_building": "extractor",
}

# Which ``MicroDuelConfig`` attribute prices each economy kind.
_KIND_COST_ATTR: dict[str, str] = {
    "build_extractor": "extractor_cost",
    "build_structure": "factory_cost",
    "produce_unit": "tank_cost",
    "upgrade_building": "extractor_upgrade_cost",
}


def _reported_capabilities(state: dict, own: list[dict]) -> set[str] | None:
    """Unit types the bridge says our builders can queue.

    ``None`` means "the bridge did not report capabilities" (offline mock engine,
    older bridge, hand-written test state). Callers must fall back to the
    funding/builder gate in that case: refusing to mask on missing information
    would disable the economy on every non-reporting environment.
    """

    builders = state.get("builders")
    if not isinstance(builders, list) or not builders:
        return None
    reported = False
    allowed: set[str] = set()
    for entry in builders:
        if not isinstance(entry, dict):
            continue
        can_build = entry.get("can_build")
        if not isinstance(can_build, list):
            continue
        reported = True
        allowed |= {str(name) for name in can_build}
    return allowed if reported else None


def build_masks(state: dict, config: MicroDuelConfig = DEFAULT_CONFIG) -> dict[str, np.ndarray]:
    """Return per-head masks.

    Layout matches ``MicroDuelConfig.action_nvec``:
    kind(K), group(G), x(P), y(P), target(T), where ``K == len(ACTION_KINDS)``.
    """

    cfg = config
    own = state.get("own_entities", [])
    enemies = state.get("visible_enemy_entities", [])
    gstate = state.get("global_state", {}) or {}
    funds = float(gstate.get("funds") or 0.0)

    # ---- kind head ------------------------------------------------------
    kind_mask = np.zeros(cfg.action_nvec[0], dtype=bool)
    # Tactical orders are always selectable; only the economy kinds are gated
    # on affordability. Everything else is gated on having a live builder.
    has_builder = any(
        unit.get("is_builder") or int(unit.get("entity_type_id") or 0) == BUILDER_TYPE_ID
        for unit in own
    )
    capabilities = _reported_capabilities(state, own)
    for index, kind in enumerate(ACTION_KINDS):
        if index >= kind_mask.shape[0]:
            break
        required = _KIND_REQUIRED_BUILDABLE.get(kind)
        buildable = (
            required is None or capabilities is None or required in capabilities
        )
        cost_attr = _KIND_COST_ATTR.get(kind)
        affordable = (
            float(getattr(cfg, cost_attr, 0.0)) <= funds if cost_attr else True
        )
        if kind == "build_extractor":
            kind_mask[index] = has_builder and buildable and affordable
        elif kind == "build_structure":
            kind_mask[index] = has_builder and buildable and affordable
        elif kind == "produce_unit":
            kind_mask[index] = has_builder and buildable and affordable
        elif kind == "upgrade_building":
            kind_mask[index] = has_builder and buildable and affordable
        elif kind == "repair_target":
            kind_mask[index] = has_builder
        else:
            kind_mask[index] = True
    if not kind_mask.any():
        # Never emit an all-false head: fall back to the tactical orders.
        kind_mask[: min(5, kind_mask.shape[0])] = True

    # ---- group head -----------------------------------------------------
    group_mask = np.zeros(cfg.groups + 1, dtype=bool)
    group_ids_known = any(unit.get("group_id") is not None for unit in own)
    if group_ids_known:
        for unit in own:
            group_mask[int(unit.get("group_id") or 0) % cfg.groups] = True
    elif own:
        # The real RWPP bridge may stream unit ids but no ``group_id``. The
        # game-side controller derives groups from ``unit.id % groups``, so all
        # groups are addressable; masking down to group 0 would silently
        # disable 75% of the army.
        group_mask[: cfg.groups] = True
    group_mask[-1] = True  # broadcast to every alive group

    # ---- position / target heads ---------------------------------------
    x_mask = np.ones(cfg.position_grid, dtype=bool)
    y_mask = np.ones(cfg.position_grid, dtype=bool)
    target_mask = np.zeros(cfg.target_slots, dtype=bool)
    for slot in range(min(len(enemies), cfg.target_slots - 1)):
        target_mask[slot] = True
    # ``build_extractor`` also reads the target head: there it selects a resource
    # pool. With only enemy slots open the policy could not express "use pool 5"
    # when the map has more pools than visible enemies, so the extractor choice
    # collapsed towards slot 0. Open the pool slots as well; the shared head is
    # unambiguous because the chosen kind decides which meaning applies.
    # An ``attack_target`` that lands on a pool-only slot is a harmless no-op
    # (``decode_action`` falls back to ``hold``), exactly like the "no target"
    # slot has always been.
    pool_slots = state.get("resource_slots") or []
    for slot in range(min(len(pool_slots), cfg.target_slots - 1)):
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
