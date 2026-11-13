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
from ..mock_engine import ACTION_KINDS

__all__ = ["build_masks", "flat_mask_vector", "masks_as_lists"]

# Which ``info.builders[].can_build`` entry each *builder* economy kind needs.
# The real game decides what a builder may queue; the mask has to agree with it,
# or the policy learns "these kinds are selectable" and then pays
# ``invalid_penalty`` on every attempt the engine rejects. On stock skirmish maps
# the built-in ``builder`` reports only ``landFactory``, so extractor and factory
# stay masked until the map actually offers them.
#
# ``produce_unit`` is deliberately *not* in this table: it is a **factory**
# action, not a builder action. Gating it on the builder's build menu made it
# permanently unreachable -- measured on Lake, ``produce_unit`` stayed masked at
# reset *and* after a landFactory had been standing for 1800 ticks, because
# ``"tank" not in builder.can_build`` (that map's builder menu is
# ``['landFactory']``). The policy could therefore never queue a tank at all, no
# matter what it built, which is exactly the "never builds anything" symptom.
_KIND_REQUIRED_BUILDABLE: dict[str, str] = {
    "build_extractor": "extractor",
    "build_structure": "landFactory",
}

# Which ``info.factories[].can_produce`` entry ``produce_unit`` needs.
#
# Deliberately empty: there is **no** unit name that is universally producible.
# Measured on Lake, a ``landFactory`` offers ``['builder', 'hoverTank']`` -- no
# ``tank`` at all -- while the base ``commandCenter`` offers only ``['builder']``.
# Requiring a hard-coded ``"tank"`` therefore kept ``produce_unit`` masked on the
# very map being trained, even with a finished factory standing. The honest gate
# is "some factory reports a non-empty menu (or reports nothing at all)".
_KIND_REQUIRED_PRODUCIBLE: dict[str, str] = {}

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


def _reported_producibles(state: dict) -> set[str] | None:
    """Unit types our *factories* say they can produce, or ``None`` if unreported."""

    factories = state.get("factories")
    if not isinstance(factories, list) or not factories:
        return None
    reported = False
    allowed: set[str] = set()
    for entry in factories:
        if not isinstance(entry, dict):
            continue
        can_produce = entry.get("can_produce")
        if not isinstance(can_produce, list):
            continue
        reported = True
        allowed |= {str(name) for name in can_produce}
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
    #
    # Only ``is_builder`` is trusted. The old fallback
    # ``entity_type_id == BUILDER_TYPE_ID(2)`` was meant for bridges that omit
    # ``is_builder``, but the real bridge used to report ``entity_type_id`` as
    # "2 for any building", so a lone command center made every economy kind
    # selectable. The policy then picked ``build_extractor`` with no builder on
    # the map, the engine rejected it, and it paid ``invalid_penalty`` every
    # decision -- the dominant negative term in the measured run. A unit that
    # cannot queue builds must not be treated as one that can, so the type-id
    # guess is gone; a bridge that omits ``is_builder`` simply leaves the
    # economy masked instead of inviting guaranteed rejections.
    has_builder = any(unit.get("is_builder") for unit in own)
    # ``upgrade_building`` needs something that is *already built* -- the
    # builder's build menu is irrelevant to it. Gating it on the builder's
    # capability (the old ``"extractor" in can_build`` requirement) both masked a
    # legal order when the map cannot build extractors yet and left it selectable
    # when nothing upgradable existed. The bridge reports the engine's own answer
    # (``can_upgrade``, from ``getBuildActionId``); when it does not, fall back to
    # "we have a building and a builder", which at worst yields one rejected
    # order instead of a permanently dead or always-invalid kind.
    if any("can_upgrade" in unit for unit in own):
        has_upgradable = any(unit.get("can_upgrade") for unit in own)
    else:
        has_upgradable = has_builder and any(unit.get("is_building") for unit in own)
    capabilities = _reported_capabilities(state, own)
    # ``produce_unit`` is gated on a *factory*, not on the builder's menu. The
    # two are unrelated capabilities: on a stock skirmish map the builder's menu
    # is ``['landFactory']`` (no ``tank``), while the factory is exactly what can
    # make tanks. Binding production to the builder made the kind permanently
    # unreachable and the agent could never field a combat unit.
    factories = [
        unit
        for unit in own
        if unit.get("is_factory") or unit.get("is_factory_building")
    ]
    producibles = _reported_producibles(state)
    # A reported but *empty* menu means the engine itself says "this factory can
    # queue nothing right now" (still under construction, tech-locked). That is a
    # real answer and must mask the kind; only unreported capabilities fall back
    # to "we own a factory".
    producible = producibles is None or len(producibles) > 0
    for index, kind in enumerate(ACTION_KINDS):
        if index >= kind_mask.shape[0]:
            break
        required = _KIND_REQUIRED_BUILDABLE.get(kind)
        buildable = (
            required is None or capabilities is None or required in capabilities
        )
        needed_unit = _KIND_REQUIRED_PRODUCIBLE.get(kind)
        if needed_unit is not None:
            producible = producibles is None or needed_unit in producibles
        cost_attr = _KIND_COST_ATTR.get(kind)
        affordable = (
            float(getattr(cfg, cost_attr, 0.0)) <= funds if cost_attr else True
        )
        if kind == "build_extractor":
            kind_mask[index] = has_builder and buildable and affordable
        elif kind == "build_structure":
            kind_mask[index] = has_builder and buildable and affordable
        elif kind == "produce_unit":
            kind_mask[index] = bool(factories) and producible and affordable
        elif kind == "upgrade_building":
            kind_mask[index] = has_upgradable and affordable
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
