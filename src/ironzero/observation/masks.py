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
from .menus import build_menu, produce_options, upgradable_buildings

__all__ = [
    "MULTI_MASK_KEYS",
    "build_masks",
    "build_masks_for_config",
    "build_masks_multi",
    "flat_mask_vector",
    "masks_as_lists",
    "multi_mask_keys",
]

# Which ``info.builders[].can_build`` entry each *builder* economy kind needs.
# The real game decides what a builder may build; the mask has to agree with it,
# or the policy learns "these kinds are selectable" and then pays
# ``invalid_penalty`` on every attempt the engine rejects.
#
# ``build_structure`` is deliberately **absent** now. It used to require the
# literal name ``landFactory``, which hard-coded both the *existence* of one
# particular building and the idea that ``build_structure`` means exactly that
# building. Neither is true any more: the kind now names a structure type through
# the target head (see ``menus.build_menu``), so the honest gate is "the reported
# structure menu is non-empty", which is evaluated below. Keeping the old entry
# would have masked the kind on any map whose builder offers a turret but no
# factory while leaving it open when the reverse held.
#
# ``produce_unit`` was never in this table: it is a **factory** action, not a
# builder action. Gating it on the builder's build menu made it permanently
# unreachable -- measured on Lake, ``produce_unit`` stayed masked at reset *and*
# after a landFactory had been standing for 1800 ticks, because ``"tank" not in
# builder.can_build`` (that map's builder menu is ``['landFactory']``). The policy
# could therefore never queue a tank at all, no matter what it built, which is
# exactly the "never builds anything" symptom.
_KIND_REQUIRED_BUILDABLE: dict[str, str] = {
    "build_extractor": "extractor",
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
    #
    # Read through ``menus.upgradable_buildings`` so the *decode* selects from the
    # identical list: the target head indexes that list, and a mask sized from a
    # different one would open slots the decode maps elsewhere.
    upgradable_units, upgrade_reported = upgradable_buildings(state)
    if upgrade_reported:
        has_upgradable = bool(upgradable_units)
    else:
        has_upgradable = has_builder and any(unit.get("is_building") for unit in own)
    capabilities = _reported_capabilities(state, own)
    # The *economy* kinds are the curriculum's job while it is running.
    #
    # Closing the builder's group is not enough: ``build_extractor`` and
    # ``build_structure`` decode into orders that address the builder directly
    # (``decoding._builder_group`` deliberately returns the broadcast slot so a
    # build order always reaches it). So the policy can still cancel construction
    # through the *kind* head, and measured it does: with the group head protected
    # the episode still ended at ``occupied = 1``, ``income = 26``, three entities
    # (``.runtime/mask_reach.json``, 1200 decisions, 0 samples from a closed group).
    #
    # Duplicating the curriculum's own decision adds nothing -- the stage already
    # issues exactly these two kinds -- while every duplicate order resets the
    # build (``.runtime/ab_reissue.py``: ``build_progress`` pinned at 0.0 vs 1.0 at
    # decision 140). So while ``protect_builder`` is set and the curriculum is
    # enabled, those kinds are masked off and the agent keeps the decisions the
    # curriculum does *not* make: what to produce, where to move, whom to attack.
    #
    # ``produce_unit`` and ``upgrade_building`` stay open -- they are not the
    # curriculum's exclusive job (the stage sustains a floor, the policy may exceed
    # it) and they do not interrupt the builder.
    #
    # ``build_structure`` is **narrowed, not closed**.
    #
    # Closing it outright was the right call while it could only ever mean "a second
    # land factory", but it stopped being right the moment the kind started naming a
    # *structure type*: the policy is now the only thing that can decide to build a
    # turret, an air factory, a repair bay or a second factory at a chosen spot, and
    # the curriculum deliberately never orders any of those (it builds the one
    # factory its production stage needs and nothing else). Permanently masking the
    # kind therefore made "never builds turrets / never builds anything but one
    # factory" a property of the *action space*, not of the learning.
    #
    # The hazard the old rule protected against is unchanged and is handled exactly:
    # a duplicate build order resets construction (``.runtime/ab_reissue.py``), and
    # the only window where a *duplicate* is possible is while our own structure is
    # unfinished -- which is precisely what ``construction_guard`` tests. Outside
    # that window there is no in-flight build to cancel, so ordering a new structure
    # is a fresh decision rather than an interruption.
    #
    # ``build_extractor`` stays owned by the curriculum: it is the one kind whose
    # duplicate is catastrophic (the extractor never finishes and every downstream
    # stage is gated on it), and the curriculum already expands the economy on a
    # stall gate.
    economy_owned_by_curriculum = bool(
        getattr(cfg, "protect_builder", True)
        and getattr(cfg, "curriculum_owns_build_kinds", True)
    )
    # While a structure is under construction, movement orders would cancel it.
    #
    # Group-level protection cannot solve this on its own, and the reason is
    # structural: the bridge assigns groups with ``unit.id % groups``, so the builder
    # (id 24 -> group 0) and the very first produced tank (id 40 -> group 0) land in
    # the **same** group. Closing "the builder's group" therefore either protects
    # nothing or disables half the army -- measured both ways
    # (``.runtime/mask_reach.json``): with the numbered group closed the policy
    # sampled the broadcast slot instead, and with the broadcast slot closed too the
    # economy still stalled at ``occupied = 1``.
    #
    # Movement is what cancels construction, so movement is what gets masked -- but
    # *only while something is unfinished*. A structure needs a few hundred ticks and
    # the rest of the episode is free, so the agent keeps nearly all of its tactical
    # decisions and loses only the ones that would destroy its own economy.
    unfinished_structures = any(
        unit.get("is_building")
        and "is_operational" in unit
        and not unit.get("is_operational")
        and float(unit.get("build_progress") or 0.0) < 1.0
        for unit in own
    )
    construction_guard = bool(
        getattr(cfg, "protect_builder", True) and unfinished_structures
    )
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
    # The (factory, unit) pairs the decoder indexes with the target head. Using the
    # same helper here and in ``rwpp.decoding`` is what guarantees "the mask opened
    # it" and "the decode can name it" are the same statement.
    options = produce_options(state)
    if options:
        producible = True
    # The structure menu is what ``build_structure`` can actually express now, so
    # it is what the kind is gated on.
    #
    # An empty menu is read against ``capabilities``: when the payload reported a
    # builder menu but everything in it was pool-only (a map where the builder can
    # make extractors and nothing else), the kind genuinely cannot express
    # anything and is masked. When ``capabilities is None`` the bridge said nothing
    # at all -- mock engine, older bridge, hand-written fixture -- and the
    # historical permissive behaviour must stay, or the economy disappears on
    # every non-reporting environment.
    structures = build_menu(state)
    structure_selectable = capabilities is None or bool(structures)
    # A menu that offers *only* ``builder`` must not open ``produce_unit``. Every
    # factory in Rusted Warfare can make builders, so gating on "the menu is
    # non-empty" let the policy spend the entire economy repeatedly queueing
    # builders -- measured as the dominant rejection in a real run
    # (``PRODUCE_UNAVAILABLE=66`` per update) and, before that, as 12 builders and
    # zero combat units on an 8000-tick Lake episode. ``produce_unit`` exists to
    # add *fighting* power; if no factory can contribute that, the kind is a trap
    # and must stay masked.
    if producibles is not None:
        producible = any(str(name).lower() != "builder" for name in producibles)
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
            kind_mask[index] = has_builder and structure_selectable and affordable
        elif kind == "produce_unit":
            kind_mask[index] = bool(factories) and producible and affordable
        elif kind == "upgrade_building":
            kind_mask[index] = has_upgradable and affordable
        else:
            kind_mask[index] = True
        if economy_owned_by_curriculum and kind == "build_extractor":
            kind_mask[index] = False
        # A duplicate build while something of ours is under construction restarts
        # it, so the kind is closed exactly in that window -- and only then. See the
        # long note at ``economy_owned_by_curriculum`` for why this is narrower than
        # the old "curriculum owns both build kinds, always".
        #
        # ``build_extractor`` belongs in this guard and did not used to be in it. The
        # gap was invisible while the curriculum owned the kind (it issues the order
        # itself, on a stall gate), but in ``curriculum_mode="mask"`` the policy owns
        # the kind and nothing else stops it from re-ordering every decision -- which
        # restarts the construction every decision. Measured with a trained
        # two-slot checkpoint on Lake (`curriculum_mode="mask"`, difficulty 1):
        # ``build_extractor`` was dispatched on **2006 of 2006** decisions, the first
        # extractor finished only at tick 16011 of 20056, no factory was ever built,
        # 11958 credits went unspent, and the enemy grew to 33 units
        # (``.runtime/policy_attrition.json``).
        #
        # Offering a kind is a promise that acting on it does something useful. A
        # re-order that cancels its own progress breaks that promise, so it is a mask
        # honesty bug rather than a strategy the policy should have to discover.
        if construction_guard and kind in {"build_structure", "build_extractor"}:
            kind_mask[index] = False
        if construction_guard and kind in {
            "move_group",
            "attack_move",
            "attack_target",
            "stop",
            "hold",
        }:
            # These are the kinds that re-task units during an active build.
            kind_mask[index] = False
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
    # Protect the builder from the policy's tactical orders.
    #
    # Every real order cancels the in-flight one, and a tactical order addressed to
    # the builder's group therefore stops whatever the builder was constructing.
    # Measured (``.runtime/ppo_interference.json``): a policy that issues one order
    # per decision sends it while a structure is unfinished on **1949 of 1949**
    # decisions, and the episode ends with ``occupied = 1``, ``income = 26`` and
    # three own entities -- against ``occupied = 3-4``, ``income = 42-50`` and a 93%
    # win rate for a policy that issues **nothing**. End to end that was 43.75% for
    # the trained policy versus 93.75% for noop: the policy was destroying the
    # economy that produces its own army.
    #
    # A group that contains only builders is therefore closed to the policy, and so
    # is the broadcast slot when every living unit is a builder. Groups that mix
    # builders with combat units stay open: closing them would disable a share of
    # the army, and the curriculum's own order rides along after the policy's in the
    # same dispatch, so a mixed group costs resumption latency rather than the build.
    #
    # This removes no real decision: "walk the builder into the enemy base" is not a
    # strategy the agent is being asked to learn here -- it is the exact failure the
    # earlier rounds documented ("the builder finishes the extractor, then charges
    # the enemy base and dies").
    if getattr(cfg, "protect_builder", True) and own:
        groups = max(1, int(cfg.groups))
        builder_groups: set[int] = set()
        combat_groups: set[int] = set()
        for unit in own:
            raw_id = unit.get("entity_id", 0)
            try:
                entity_id = int(raw_id)
            except (TypeError, ValueError):
                entity_id = 0
            group = unit.get("group_id")
            if group is None:
                group = entity_id % groups
            try:
                group_index = int(group) % groups
            except (TypeError, ValueError):
                group_index = 0
            if unit.get("is_builder"):
                builder_groups.add(group_index)
            else:
                combat_groups.add(group_index)
        # Only act when the payload actually distinguishes the two roles. A state
        # with no ``is_builder`` flag at all (the mock engine, hand-written test
        # fixtures, an older bridge) would otherwise classify *every* unit as a
        # combatant and close nothing -- or, worse, classify nothing as a combatant
        # and close the broadcast slot for no reason.
        if builder_groups:
            builder_only = builder_groups - combat_groups
            for group_index in builder_only:
                group_mask[group_index] = False
            # The broadcast slot orders *every* living unit, builder included.
            #
            # This must be closed whenever the builder lives in a group the policy
            # may not address, **even if other combat units exist**: the broadcast
            # slot reaches the builder just as surely as its own numbered group does.
            # Measured (``.runtime/mask_reach.json``): with only the numbered group
            # closed, the policy sampled the broadcast slot on 384-423 of 1200
            # decisions, every ``attack_move`` through it cancelled the extractor,
            # and the episode still ended at ``occupied = 1`` / ``income = 26`` with
            # three own entities.
            #
            # The cost is real but small: the policy loses "order the whole army at
            # once" and must address groups individually. The benefit is that the
            # builder -- the single unit that produces everything else -- is no
            # longer reachable by a movement order.
            group_mask[-1] = False
            if not group_mask.any():
                # Never emit an all-false head (mirrors the kind-head guard above).
                # Falling back to the broadcast slot keeps sampling valid; the worst
                # case is the pre-existing behaviour.
                group_mask[-1] = True

    # ---- position / target heads ---------------------------------------
    x_mask = np.ones(cfg.position_grid, dtype=bool)
    y_mask = np.ones(cfg.position_grid, dtype=bool)
    target_mask = np.zeros(cfg.target_slots, dtype=bool)
    for slot in range(min(len(enemies), cfg.target_slots - 1)):
        target_mask[slot] = True
    # Open the slots that name a *menu entry* for the economy kinds.
    #
    # The target head is shared by four unrelated meanings, and until now only two
    # of them (enemy slot, resource pool) were ever opened. The other two were
    # therefore unreachable *by construction*:
    #
    #   * ``produce_unit``  -> an entry of ``menus.produce_options`` (a
    #     (factory, unit-type) pair);
    #   * ``upgrade_building`` -> an entry of ``menus.upgradable_buildings``;
    #   * ``build_structure``  -> an entry of ``menus.build_menu`` (a structure
    #     type).
    #
    # Measured consequence of leaving them closed: the decode could only ever fall
    # back to its default constant for each kind -- one structure type
    # (``land_factory``), one unit type (the menu's first combat entry), one
    # building to upgrade. That is precisely the reported "only builds one kind of
    # tank", "never builds turrets", and "never upgrades". The capability existed
    # in the engine and in the bridge payload the whole time
    # (``.runtime/buildnames.json``: ``turret`` is accepted on this map); only the
    # head that names it was shut.
    #
    # Sizing: the head has ``cfg.target_slots`` entries shared with the enemy
    # slots. Opening ``min(len(menu), target_slots - 1)`` keeps the "no target"
    # fallback and never exceeds the head, and the decoder uses the same modulo so
    # a menu longer than the head still resolves every slot to a real entry.
    #
    # This is additive to the historical behaviour: an environment whose payload
    # reports no menus (mock engine, older bridge, hand-built fixtures) gets exactly
    # the previous mask, because each helper returns an empty list there.
    economy_menu_sizes = (
        len(produce_options(state)),
        len(upgradable_buildings(state)[0]),
        len(build_menu(state)),
    )
    widest = max(economy_menu_sizes)
    for slot in range(min(widest, cfg.target_slots - 1)):
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


#: Mask keys of the two-slot layout, in the exact order
#: :func:`ironzero.rwpp.decoding.multi_action_nvec` lays the heads out: economy
#: slot first, tactical slot second.
#:
#: The keys are **prefixed** rather than reused. Reusing ``kind``/``group``/``x``
#: for both slots would let a single-slot caller silently consume a two-slot mask
#: (and vice versa) and, worse, would make ``masks_to_tensors`` stack a 5-key and a
#: 10-key dict without complaint because the key names happen to overlap. Distinct
#: names turn that class of bug into a ``KeyError`` at the seam.
MULTI_MASK_KEYS: tuple[str, ...] = (
    "econ_kind",
    "econ_group",
    "econ_x",
    "econ_y",
    "econ_target",
    "tac_kind",
    "tac_group",
    "tac_x",
    "tac_y",
    "tac_target",
)


def multi_mask_keys() -> tuple[str, ...]:
    """The 10 mask keys of the two-slot layout, in action-head order."""

    return MULTI_MASK_KEYS


def _single_slot_view(config: MicroDuelConfig) -> MicroDuelConfig:
    """``config`` with the single-slot action layout, for reuse of ``build_masks``.

    ``build_masks`` sizes its kind head from ``config.action_nvec[0]``. Under a
    two-slot config that value is ``len(ECONOMY_ACTION_KINDS) == 5``, so asking it
    for the single-slot masks would return a **5-wide** kind mask indexed by
    economy-slot positions instead of the 10-wide ``ACTION_KINDS`` mask every other
    consumer expects -- and the first ``ACTION_KINDS.index("build_extractor")``
    lookup would either be wrong or out of range (measured: ``IndexError: index 5
    is out of bounds for axis 0 with size 5``).

    Everything else (group / position / target widths, entity limits, fog) is
    identical between the two layouts by construction, so swapping only
    ``action_slots`` yields exactly the single-slot legality set the two-slot
    masks must be a restriction of.
    """

    if int(getattr(config, "action_slots", 1)) == 1:
        return config
    from dataclasses import replace as _replace

    return _replace(config, action_slots=1)


def build_masks_multi(
    state: dict, config: MicroDuelConfig = DEFAULT_CONFIG
) -> dict[str, np.ndarray]:
    """Per-head masks for the two-slot (economy + tactical) action layout.

    Why a separate function instead of a flag on :func:`build_masks`
    -----------------------------------------------------------------
    The single-slot masks answer "which of the 10 kinds may this decision pick?".
    The two-slot masks must answer that *twice*, against two different
    vocabularies, and the answer to one no longer constrains the other: an
    economy slot choosing ``build_extractor`` and a tactical slot choosing
    ``attack_move`` are both legal **in the same decision**, which the single flat
    vector could not express at all (it had to pick one kind).

    Construction is therefore "restrict, never invent": this function calls
    :func:`build_masks` once and projects the result onto each slot's vocabulary.
    A kind the single-slot mask closes stays closed in *both* slots, so splitting
    the head cannot release an order the engine would reject -- the invariant
    ``econ_kind[i] implies kind[i]`` (and the same for ``tac_kind``) is the whole
    justification for reusing the existing gate logic instead of duplicating it.

    The group / position / target heads are shared across slots because both slots
    address the same map with the same quantisation: the economy slot names a
    build position and a structure type, the tactical slot names a move position
    and a target. Keeping one set of masks means "target slot 3" and "group 4"
    mean the same thing in both slots and in the single-slot layout, so a mask
    explained for one layout still explains the other.

    Empty entries
    -------------
    Each slot can always say "nothing this decision", and that entry is
    **unconditionally open**. It is the mechanism by which a decision chooses to
    leave the economy alone (so construction is not cancelled) while still
    commanding the army, and vice versa. Closing it would force the policy to
    issue *some* economy order every decision, which is exactly the
    cancel-construction failure this layout exists to remove.

    Raises ``ValueError`` when ``config.action_slots != 2``: producing these masks
    for a single-slot config would hand the policy 10 heads' worth of masks for a
    5-head network, and the resulting mismatch fails deep inside PPO.
    """

    if not getattr(config, "is_two_slot", False):
        raise ValueError(
            "build_masks_multi requires MicroDuelConfig.action_slots == 2 "
            f"(got {getattr(config, 'action_slots', None)!r}); "
            "use build_masks for the single-slot layout"
        )

    # ``rwpp.decoding`` imports this package, so the vocabulary is imported here
    # rather than at module scope to avoid a cycle.
    from ..rwpp.decoding import (
        ECONOMY_ACTION_KINDS,
        ECONOMY_EMPTY_INDEX,
        TACTICAL_ACTION_KINDS,
        TACTICAL_EMPTY_INDEX,
    )

    single = build_masks(state, _single_slot_view(config))

    def restrict(vocabulary: tuple[str, ...], empty_index: int) -> np.ndarray:
        """The single-slot kind mask projected onto one slot's vocabulary."""

        allowed = np.zeros(len(vocabulary), dtype=bool)
        for position, name in enumerate(vocabulary):
            if position == empty_index:
                # "Do nothing in this slot" is always expressible. It emits no
                # order at all, so it can never cancel an in-flight construction.
                allowed[position] = True
                continue
            try:
                kind_index = ACTION_KINDS.index(name)
            except ValueError:  # pragma: no cover - vocabulary/kind table drift
                continue
            allowed[position] = bool(single["kind"][kind_index])
        return allowed

    return {
        "econ_kind": restrict(ECONOMY_ACTION_KINDS, ECONOMY_EMPTY_INDEX),
        "econ_group": single["group"].copy(),
        "econ_x": single["x"].copy(),
        "econ_y": single["y"].copy(),
        "econ_target": single["target"].copy(),
        "tac_kind": restrict(TACTICAL_ACTION_KINDS, TACTICAL_EMPTY_INDEX),
        "tac_group": single["group"].copy(),
        "tac_x": single["x"].copy(),
        "tac_y": single["y"].copy(),
        "tac_target": single["target"].copy(),
    }


def build_masks_for_config(
    state: dict, config: MicroDuelConfig = DEFAULT_CONFIG
) -> dict[str, np.ndarray]:
    """Layout-aware mask entry point: the one callers should use.

    The environment builds masks in two places (``RwppDuelEnv.reset`` and
    ``stepping.absorb_message``) and both must agree with the action layout the
    policy is running. Scattering ``if config.is_two_slot`` across those call sites
    is how the two drift apart -- one updated, the other forgotten -- and the
    symptom is subtle rather than loud: the policy samples from a 5-head mask
    dict while owning 10 heads, ``masks_to_tensors`` raises ``KeyError`` deep
    inside the training loop, or (worse) a stale single-slot dict is silently
    reused for a two-slot network.

    So the dispatch lives here, next to the two implementations it chooses
    between, and it returns a dict whose keys match ``config.action_nvec``:

    * single slot  -> ``{"kind", "group", "x", "y", "target"}`` (5 keys);
    * two slots    -> :data:`MULTI_MASK_KEYS` (10 keys, economy block first).

    Adding a third layout means changing this function and ``action_nvec``
    together, which is exactly the coupling that should be visible in one place.
    """

    if getattr(config, "is_two_slot", False):
        return build_masks_multi(state, config)
    return build_masks(state, config)


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
