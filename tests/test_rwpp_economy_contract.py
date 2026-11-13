# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the economy bridge contract.

These lock the three defects that made "the agent never learns to build" look
like a training problem when it was a plumbing problem:

1. **Resource pools are terrain, not units.** The bridge used to scan
   ``Unit.unitList`` for resource entities and always came back empty (ore lives
   in ``Map.resourcePoolPoints``), so ``build_extractor`` aimed at map (0,0) and
   every order was rejected.
2. **The build tech level is not 0.** ``Waypoint.setBuildTo(..., 0)`` never
   matched a ``SpecialActionBuild`` (those are techLevel 1), so
   ``isValidNewWaypoint`` rejected every build.
3. **``bonus_build`` was unreachable and then explosive.** The term tested
   ``receipt["kind"]``, which ``ReceiptMessage`` does not carry, so it was
   permanently 0.0; the hard-coded replacement value (+5.0) would have outpaid a
   won game (+1.0).

The Java side is covered by ``scripts/rwpp_econ_smoke.py`` (it needs a real
game). Everything asserted here is the Python/Python-visible contract.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from ironzero.mock_engine import ACTION_KINDS
from ironzero.observations import encode_observation
from ironzero.rwpp.config import RwppGameConfig
from ironzero.rwpp.reward_shaping import compute_step_shaping

REPO_ROOT = Path(__file__).resolve().parents[1]
BRIDGE_SOURCE = REPO_ROOT / "src" / "ironzero_bridge" / "java" / "ironzero" / "IronZeroBridge.java"


def _shaping(
    *,
    actions: list[dict],
    receipts: list[dict],
    income: float = 0.0,
    previous_income_rate: float = 0.0,
    own_buildings: int = 0,
    previous_own_buildings: int = 0,
):
    """Run one shaping step with just enough state to exercise the terms."""

    micro = RwppGameConfig().micro_config()
    game = RwppGameConfig()
    own = [{"x": 100.0, "y": 100.0, "hp": 100.0, "max_hp": 100.0}]
    own += [
        {"x": 100.0, "y": 100.0, "hp": 100.0, "max_hp": 100.0, "is_building": True}
        for _ in range(max(0, own_buildings))
    ]
    state = {
        "tick": 100,
        "own_entities": own,
        "visible_enemy_entities": [{"x": 200.0, "y": 100.0, "hp": 100.0, "max_hp": 100.0}],
        "global_state": {"income": income, "funds": 1000.0},
    }
    return compute_step_shaping(
        micro=micro,
        game=game,
        state=state,
        actions=actions,
        receipts=receipts,
        message={"global_state": {"income": income}},
        bridge_result=None,
        potential=0.0,
        previous_potential=0.0,
        distance=0.5,
        previous_distance=0.5,
        own_alive=1,
        enemy_alive=1,
        previous_own_alive=1,
        previous_enemy_alive=1,
        strategy_component=0.0,
        disconnected=False,
        done=False,
        own_max_ref=210.0,
        enemy_max_ref=210.0,
        episode_components={},
        previous_income_rate=previous_income_rate,
        previous_own_buildings=previous_own_buildings,
    )


# ---------------------------------------------------------------------------
# 1. Build reward is paid for a finished building, bounded, and below a win.
# ---------------------------------------------------------------------------
def test_completed_building_pays_the_configured_bonus() -> None:
    """A finished structure must actually earn reward."""

    shaping = _shaping(
        actions=[{"kind": "build_extractor", "unit_group": 4, "slot_id": 0}],
        receipts=[{"accepted": True}],
        own_buildings=1,
        previous_own_buildings=0,
    )
    assert shaping.components["bonus_build"] == pytest.approx(
        RwppGameConfig().build_bonus
    )
    assert shaping.components["bonus_build"] > 0.0


def test_repeating_an_accepted_order_cannot_farm_the_build_bonus() -> None:
    """Re-issuing a build order must not keep paying.

    Paying per accepted receipt let a policy earn +0.2 every decision just by
    re-queueing the same extractor; the measured exploit paid +12.0 over a
    60-decision episode.
    """

    farming = [
        _shaping(
            actions=[{"kind": "build_extractor", "unit_group": 4, "slot_id": 0}],
            receipts=[{"accepted": True}],
            # No new building appears: the order only re-queues.
            own_buildings=0,
            previous_own_buildings=0,
        )
        for _ in range(20)
    ]
    assert all(step.components["bonus_build"] == 0.0 for step in farming)


def test_two_new_buildings_pay_twice() -> None:
    shaping = _shaping(
        actions=[],
        receipts=[],
        own_buildings=3,
        previous_own_buildings=1,
    )
    assert shaping.components["bonus_build"] == pytest.approx(
        2.0 * RwppGameConfig().build_bonus
    )


def test_losing_a_building_does_not_pay() -> None:
    shaping = _shaping(
        actions=[],
        receipts=[],
        own_buildings=0,
        previous_own_buildings=2,
    )
    assert shaping.components["bonus_build"] == 0.0


def test_build_bonus_stays_below_the_win_reward() -> None:
    """Farming extractors must never outpay winning the game."""

    game = RwppGameConfig()
    micro = game.micro_config()
    assert game.build_bonus < micro.win_reward
    assert game.build_bonus <= 0.25 * micro.win_reward


# ---------------------------------------------------------------------------
# 2. Income pays for growth, not for the standing rate.
# ---------------------------------------------------------------------------
def test_steady_income_yields_no_reward_after_the_first_step() -> None:
    """A level income rate must not accumulate reward every decision."""

    steady = _shaping(
        actions=[],
        receipts=[],
        income=18.0,
        previous_income_rate=18.0,
    )
    assert steady.components["income"] == 0.0


def test_new_extractor_pays_once_as_income_growth() -> None:
    grown = _shaping(
        actions=[],
        receipts=[],
        income=18.0,
        previous_income_rate=0.0,
    )
    assert grown.components["income"] > 0.0
    # And it does not repeat: the next step with the same rate is flat again.
    repeat = _shaping(
        actions=[],
        receipts=[],
        income=18.0,
        previous_income_rate=18.0,
    )
    assert repeat.components["income"] == 0.0


# ---------------------------------------------------------------------------
# 3. The Java bridge reads ore from the map, not from the unit list.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not BRIDGE_SOURCE.is_file(), reason="bridge source not present")
def test_bridge_reads_resource_pools_from_the_map() -> None:
    """``resourceSlots`` must use ``Map.resourcePoolPoints``."""

    source = BRIDGE_SOURCE.read_text(encoding="utf-8", errors="replace")
    method = _method_body(source, "resourceSlots")
    assert "resourcePoolPoints" in method, (
        "resourceSlots must read Map.resourcePoolPoints: ore is terrain in "
        "Rusted Warfare, not a unit, so scanning Unit.unitList finds nothing"
    )
    assert "isResourceUnit" not in method


@pytest.mark.skipif(not BRIDGE_SOURCE.is_file(), reason="bridge source not present")
def test_bridge_uses_the_action_tech_level_for_builds() -> None:
    """Build waypoints must carry the action's tech level, never a literal 0."""

    source = BRIDGE_SOURCE.read_text(encoding="utf-8", errors="replace")
    method = _method_body(source, "buildWithBuilders")
    assert "getTechLevel()" in method
    assert "setBuildTo(x, y, unitType, 0)" not in method
    assert "setBuildWaypoint(x, y, unitType, 0)" not in method


@pytest.mark.skipif(not BRIDGE_SOURCE.is_file(), reason="bridge source not present")
def test_bridge_implements_every_advertised_action_kind() -> None:
    """``produce_unit`` / ``upgrade_building`` must be dispatched, not dropped."""

    source = BRIDGE_SOURCE.read_text(encoding="utf-8", errors="replace")
    method = _method_body(source, "applyStep")
    for kind in ("produce_unit", "upgrade_building"):
        assert f'"{kind}".equals(kind)' in method, (
            f"{kind} falls through to INVALID_MESSAGE, so the action space "
            "advertises a capability the bridge does not have"
        )


@pytest.mark.skipif(not BRIDGE_SOURCE.is_file(), reason="bridge source not present")
def test_bridge_reports_live_income_and_resource_slots() -> None:
    """The observation must not hard-code a dead economy."""

    source = BRIDGE_SOURCE.read_text(encoding="utf-8", errors="replace")
    assert '\\"income\\":0.0' not in source, "income is hard-coded to 0.0"
    assert "resourceSlotsJson()" in source, "observation does not emit resource_slots"


# ---------------------------------------------------------------------------
# 4. Observation contract: normalisation stays in range on a real economy map.
# ---------------------------------------------------------------------------
def _economy_state(
    *,
    own: list[dict],
    enemies: list[dict],
    funds: float = 4000.0,
    income: float = 18.0,
    occupied_slots: int = 0,
) -> dict:
    return {
        "tick": 1000,
        "own_entities": own,
        "visible_enemy_entities": enemies,
        "global_state": {"funds": funds, "income": income},
        "resource_slots": [
            {"slot_id": index, "x": 100.0, "y": 100.0, "occupied": index < occupied_slots}
            for index in range(9)
        ],
    }


def test_hp_features_stay_normalised_when_army_is_buildings() -> None:
    """A 4000 HP base must not blow the HP ratio up to ~9.9.

    ``total_hp`` used to be ``units_per_team * unit_hp``; on an economy map the
    own side is a command center (4000 HP) plus a builder (170 HP), so a
    2-tank denominator (420) produced own_hp ~= 9.9 and enemy_hp ~= 15.1.
    """

    micro = RwppGameConfig().micro_config()
    state = _economy_state(
        own=[
            {"entity_id": 23, "x": 990.0, "y": 2010.0, "hp": 4000.0,
             "max_hp": 4000.0, "is_building": True, "is_factory": True},
            {"entity_id": 24, "x": 990.0, "y": 2070.0, "hp": 170.0,
             "max_hp": 170.0, "is_builder": True},
        ],
        enemies=[
            {"entity_id": 21, "x": 1230.0, "y": 210.0, "hp": 4000.0,
             "max_hp": 4000.0, "is_building": True},
        ],
    )
    vector = encode_observation(state, micro)
    own_hp_feature = float(vector[3])
    enemy_hp_feature = float(vector[4])
    assert 0.0 <= own_hp_feature <= 1.0
    assert 0.0 <= enemy_hp_feature <= 1.0
    # A full-health army should read as full, not as a small slice of it.
    assert own_hp_feature == pytest.approx(1.0)
    assert enemy_hp_feature == pytest.approx(1.0)


def test_count_features_clamp_when_enemy_reinforces() -> None:
    """Enemy count can exceed ``units_per_team`` once the AI reinforces."""

    micro = RwppGameConfig().micro_config()
    state = _economy_state(
        own=[{"entity_id": 1, "x": 0.0, "y": 0.0, "hp": 100.0, "max_hp": 100.0}],
        enemies=[
            {"entity_id": 100 + i, "x": float(i), "y": 0.0, "hp": 100.0, "max_hp": 100.0}
            for i in range(40)
        ],
    )
    vector = encode_observation(state, micro)
    assert 0.0 <= float(vector[1]) <= 1.0
    assert float(vector[2]) == pytest.approx(1.0)


def test_all_unsigned_global_features_stay_in_unit_range() -> None:
    """Every global feature except the two signed deltas is in [0, 1]."""

    micro = RwppGameConfig().micro_config()
    state = _economy_state(
        own=[
            {"entity_id": 23, "x": 990.0, "y": 2010.0, "hp": 4000.0,
             "max_hp": 4000.0, "is_building": True, "is_factory": True},
        ],
        enemies=[
            {"entity_id": 21, "x": 1230.0, "y": 210.0, "hp": 4000.0, "max_hp": 4000.0},
        ],
        funds=99999.0,
        income=9999.0,
        occupied_slots=9,
    )
    vector = encode_observation(state, micro)
    signed = {9, 10}
    for index in range(micro.global_dim):
        value = float(vector[index])
        lower = -1.0 if index in signed else 0.0
        assert lower - 1e-6 <= value <= 1.0 + 1e-6, (
            f"global feature {index} = {value:+.3f} escaped "
            f"[{lower}, 1] on a realistic economy observation"
        )


# ---------------------------------------------------------------------------
# 5. Own entity slots are prioritised so buildings survive truncation.
# ---------------------------------------------------------------------------
def test_own_slots_keep_buildings_when_budget_is_small() -> None:
    """With 8 slots, a base placed late in the stream must still be encoded."""

    from ironzero.observation.features import prioritise_own

    micro = RwppGameConfig().micro_config()
    # Spawn order puts the base and builder AFTER a pile of combat units, which
    # is exactly what happens once production is running.
    own = [
        {"entity_id": 200 + i, "x": float(i), "y": 0.0, "hp": 210.0, "max_hp": 210.0}
        for i in range(20)
    ]
    own[18] = {"entity_id": 999, "x": 990.0, "y": 2010.0, "hp": 4000.0,
               "max_hp": 4000.0, "is_building": True, "is_factory": True}
    own[19] = {"entity_id": 998, "x": 990.0, "y": 2070.0, "hp": 170.0,
               "max_hp": 170.0, "is_builder": True}

    ordered = prioritise_own(own)
    assert int(ordered[0]["entity_id"]) == 999, "factory must take the first slot"
    assert int(ordered[1]["entity_id"]) == 998, "builder must take the second slot"

    # And the encoder must actually place them inside the first 8 slots.
    state = _economy_state(own=own, enemies=[])
    vector = encode_observation(state, micro)
    offset = micro.global_dim
    first_own_entity = vector[offset : offset + micro.entity_features]
    assert float(first_own_entity[0]) == pytest.approx(1.0), "slot 0 must be occupied"
    # Slot 0 is the factory: its position is the base, not a combat unit at x=0.
    assert float(first_own_entity[1]) == pytest.approx(990.0 / micro.map_width, abs=1e-3)


def test_enemy_slots_keep_stream_order_for_stable_targeting() -> None:
    """Enemy order must not shuffle: target slot i must mean the same enemy."""

    micro = RwppGameConfig().micro_config()
    enemies = [
        {"entity_id": 500, "x": 1000.0, "y": 0.0, "hp": 100.0, "max_hp": 100.0},
        {"entity_id": 501, "x": 1.0, "y": 0.0, "hp": 100.0, "max_hp": 100.0},
    ]
    state = _economy_state(
        own=[{"entity_id": 1, "x": 0.0, "y": 0.0, "hp": 100.0, "max_hp": 100.0}],
        enemies=enemies,
    )
    vector = encode_observation(state, micro)
    _, info = None, None
    # The enemy block starts after the own block.
    enemy_offset = (
        micro.global_dim + micro.max_entities * micro.entity_features
    )
    first_enemy = vector[enemy_offset : enemy_offset + micro.entity_features]
    # Stream order preserved: entity 500 (x=1000) is slot 0 even though it is
    # farther than entity 501 (x=1), so distance sorting is NOT applied.
    assert float(first_enemy[1]) == pytest.approx(1000.0 / micro.map_width, abs=1e-3)


# ---------------------------------------------------------------------------
# 6. Evaluation must not collide with itself on a bridge port.
# ---------------------------------------------------------------------------
def test_eval_ports_rotate_so_background_eval_cannot_collide() -> None:
    """Background and final eval used to share one port and the run died.

    The 2026-11-12 500-update run finished every update, then crashed at the
    final evaluation with ``BindException`` because the async evaluation from
    update 10 was still holding port 5200.
    """

    import threading

    from ironzero.training.rwpp_ppo import RwppPPOTrainer

    trainer = RwppPPOTrainer.__new__(RwppPPOTrainer)
    trainer.eval_port_base = 5200
    trainer.EVAL_PORT_POOL = 16
    trainer._eval_port_cursor = 0
    trainer._eval_port_lock = threading.Lock()

    # Each evaluation reserves a *block* of ports, one per parallel episode.
    ports = [trainer._reserve_eval_ports(1) for _ in range(trainer.EVAL_PORT_POOL)]
    assert len(set(ports)) == trainer.EVAL_PORT_POOL, (
        "consecutive evaluations reused a bridge port, which is what makes a "
        "background evaluation and the final evaluation collide"
    )
    assert min(ports) >= 5200
    assert trainer._reserve_eval_ports(1) == 5200  # wraps around predictably


def test_parallel_eval_blocks_never_overlap() -> None:
    """A 4-episode parallel evaluation must own 4 distinct ports.

    Regression guard: evaluation used to be sequential on one port. Making it
    parallel is only safe if the block reservation keeps two overlapping
    evaluations from sharing any port, since two games on one port die with
    ``BindException`` and take the whole run down.
    """

    import threading

    from ironzero.training.rwpp_ppo import RwppPPOTrainer

    trainer = RwppPPOTrainer.__new__(RwppPPOTrainer)
    trainer.eval_port_base = 5200
    trainer.EVAL_PORT_POOL = 16
    trainer._eval_port_cursor = 0
    trainer._eval_port_lock = threading.Lock()

    first = trainer._reserve_eval_ports(4)
    second = trainer._reserve_eval_ports(4)
    block_a = set(range(first, first + 4))
    block_b = set(range(second, second + 4))
    assert block_a == {5200, 5201, 5202, 5203}
    assert block_b == {5204, 5205, 5206, 5207}
    assert not (block_a & block_b)


def test_parallel_eval_rejects_more_episodes_than_ports() -> None:
    """Silently wrapping the cursor would hand out an in-use port."""

    import threading

    import pytest

    from ironzero.training.rwpp_ppo import RwppPPOTrainer

    trainer = RwppPPOTrainer.__new__(RwppPPOTrainer)
    trainer.eval_port_base = 5200
    trainer.EVAL_PORT_POOL = 16
    trainer._eval_port_cursor = 0
    trainer._eval_port_lock = threading.Lock()

    with pytest.raises(RuntimeError, match="eval_episodes"):
        trainer._reserve_eval_ports(17)


def test_terminal_step_does_not_book_a_phantom_income_collapse() -> None:
    """A ``done`` message has no ``global_state``; that is not "income = 0".

    Reading the missing key as 0 booked a full -rate collapse on the final
    decision (measured: -0.9 for one active extractor), which punished the agent
    for ending the episode.
    """

    from ironzero.micro_config import MicroDuelConfig
    from ironzero.rwpp.config import RwppGameConfig as GameConfig

    game = GameConfig()
    micro = game.micro_config()

    def run(message_global: dict) -> float:
        shaping = compute_step_shaping(
            micro=micro,
            game=game,
            state={
                "tick": 100,
                "own_entities": [{"x": 100.0, "y": 100.0, "hp": 100.0, "max_hp": 100.0}],
                "visible_enemy_entities": [
                    {"x": 200.0, "y": 100.0, "hp": 100.0, "max_hp": 100.0}
                ],
                "global_state": message_global,
            },
            actions=[],
            receipts=[],
            message={"global_state": message_global},
            bridge_result="timeout",
            potential=0.0,
            previous_potential=0.0,
            distance=0.5,
            previous_distance=0.5,
            own_alive=1,
            enemy_alive=1,
            previous_own_alive=1,
            previous_enemy_alive=1,
            strategy_component=0.0,
            disconnected=False,
            done=True,
            own_max_ref=210.0,
            enemy_max_ref=210.0,
            episode_components={},
            previous_income_rate=18.0,
            previous_own_buildings=1,
        )
        return shaping.components["income"]

    # Terminal observation without economy data: no delta, so no punishment.
    assert run({}) == 0.0
    # A real, reported collapse still pays.
    assert run({"income": 0.0}) < 0.0


def _method_body(source: str, name: str) -> str:
    """Return the text of a Java method body (brace-counted, best effort).

    The bridge is one large Java file with no parser available here, so this
    finds the *declaration* line (``... name(`` preceded by a return type on the
    same line) and walks braces from there. Searching for a bare ``name(`` would
    also match call sites such as ``produceUnit(...)`` inside ``applyStep``.
    """

    import re

    pattern = re.compile(
        rf"^[ \t]*(?:public|private|protected)[^\n;=]*\b{re.escape(name)}\s*\(",
        re.MULTILINE,
    )
    match = pattern.search(source)
    if match is None:
        raise AssertionError(f"method declaration not found in bridge source: {name}")
    brace = source.find("{", match.end())
    if brace < 0:
        raise AssertionError(f"method has no body: {name}")
    depth = 0
    for index in range(brace, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[brace : index + 1]
    raise AssertionError(f"unbalanced braces while reading {name}")


# ---------------------------------------------------------------------------
# 7. Rejection reasons must be observable, not just a number in the reward.
# ---------------------------------------------------------------------------
def test_receipt_reasons_histograms_only_rejections() -> None:
    """``invalid`` is one reward key; the histogram is what makes it diagnosable."""

    from ironzero.rwpp.stepping import receipt_reasons

    counts = receipt_reasons(
        [
            {"accepted": True},
            {"accepted": False, "rejected_reason": "ACTION_MASKED"},
            {"accepted": False, "rejected_reason": "ACTION_MASKED"},
            {"accepted": False, "rejected_reason": "TARGET_INVALID"},
        ]
    )
    assert counts == {"ACTION_MASKED": 2, "TARGET_INVALID": 1}
    assert receipt_reasons([]) == {}
    # A rejection with no reason is still a rejection: never silently dropped.
    assert receipt_reasons([{"accepted": False}]) == {"UNKNOWN": 1}


def test_step_info_carries_this_decisions_receipts() -> None:
    """The ``receipts`` key fed to ``build_step_info`` must exist.

    Regression guard: the rejection histogram was added to ``build_step_info``
    but ``RwppDuelEnv.step`` never passed ``receipts`` in its ``ctx``, so the
    histogram was permanently empty -- the diagnostic silently reported "no
    rejections" while the reward kept paying ``invalid_penalty``.
    """

    import inspect

    from ironzero.rwpp.env import RwppDuelEnv

    source = inspect.getsource(RwppDuelEnv.step)
    ctx_start = source.find("build_step_info(")
    assert ctx_start >= 0, "step() no longer builds step info"
    ctx = source[ctx_start:]
    assert '"receipts"' in ctx, (
        "step() must pass this decision's receipts to build_step_info, "
        "otherwise info['rejection_reasons'] is always empty"
    )
    assert "receipts_this_step" in ctx, (
        "the receipts passed must be this step's slice, not the episode total"
    )

# ---------------------------------------------------------------------------
# 5. The builder's reported menu must cover what the engine actually accepts.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not BRIDGE_SOURCE.is_file(), reason="bridge source not present")
def test_builder_menu_is_scanned_not_probed_from_a_fixed_list() -> None:
    """``buildersJson`` must report the engine's real menu, not a 4-name probe.

    Measured failure this guards: on ``[p2]Lake (2p)`` the bridge reported
    ``can_build: ["landFactory"]`` for the map-placed builder while the very same
    builder *accepted* ``structure_type="extractor"``. The Python action mask
    gates ``build_extractor`` on the literal name ``"extractor"`` appearing in
    this list, so the kind was closed for every decision of every episode and the
    policy could not express "build an extractor" at all -- the agent literally
    had no way to learn the economy, however long it trained.

    ``canQueueAction`` answers "is this in the build *menu*", which is a
    different question from "would the engine accept this build order".
    ``buildWithBuilders`` dispatches through ``getSpecialAction`` +
    ``isValidNewWaypoint``, so the menu has to be derived the same way or the
    mask and the dispatcher disagree.
    """

    source = BRIDGE_SOURCE.read_text(encoding="utf-8", errors="replace")
    method = _method_body(source, "buildersJson")
    assert "buildableNamesFor" in method, (
        "buildersJson must delegate to the registry scan; probing a fixed "
        "BUILDABLE_PROBE list reports a menu narrower than the dispatchable set"
    )
    assert "BUILDABLE_PROBE" not in method, (
        "the fixed-list probe is exactly the bug: it reported [landFactory] on a "
        "map whose builder accepts extractor orders"
    )

    scan = _method_body(source, "buildableNamesFor")
    assert "UnitType.values()" in scan, (
        "the builder menu must come from a registry scan, like the factory menu"
    )
    assert "getSpecialAction" in scan, (
        "the scan must use the same lookup buildWithBuilders dispatches with, "
        "otherwise the reported menu and the accepted orders can diverge again"
    )
    assert "CANONICAL_BUILD_NAMES" in scan, (
        "the registry scan alone misses 'extractor' on stock maps (the engine's "
        "internal name there is extractorT1), so the canonical names the action "
        "mask matches on must be probed explicitly"
    )


@pytest.mark.skipif(not BRIDGE_SOURCE.is_file(), reason="bridge source not present")
def test_canonical_build_names_cover_every_masked_economy_kind() -> None:
    """Every name the action mask gates on must be in the probe list.

    ``masks._KIND_REQUIRED_BUILDABLE`` maps ``build_extractor`` -> ``"extractor"``
    and ``build_structure`` -> ``"landFactory"``. If the bridge never reports one
    of those strings, that kind is unreachable no matter what the engine would
    accept.
    """

    from ironzero.observation.masks import _KIND_REQUIRED_BUILDABLE

    source = BRIDGE_SOURCE.read_text(encoding="utf-8", errors="replace")
    start = source.index("CANONICAL_BUILD_NAMES")
    block = source[start:source.index("};", start)]
    for kind, required in _KIND_REQUIRED_BUILDABLE.items():
        assert f'"{required}"' in block, (
            f"{kind} is masked on the name {required!r}, but the bridge's "
            f"canonical probe list does not ask for it"
        )

@pytest.mark.skipif(not BRIDGE_SOURCE.is_file(), reason="bridge source not present")
def test_unfinished_factory_reports_an_empty_production_menu() -> None:
    """A half-built factory must not advertise a menu it cannot honour.

    Measured failure this guards (``.runtime/produce_gate.json``): on
    ``[p2]Lake (2p)`` a ``landFactory`` at build progress 0.01 -- with
    ``is_operational=false`` -- reported ``can_produce: ["builder","hoverTank"]``.
    The Python action mask opens ``produce_unit`` as soon as any factory offers a
    non-``builder`` entry, so the kind was selectable hundreds of ticks before the
    engine would accept the order. Measured consequence: every early order came
    back ``PRODUCE_UNAVAILABLE`` and ``bonus_produce`` stayed at exactly 0.000
    across 200 updates, even though the same factory accepts the same order later
    in the same episode.

    An empty list is a real answer ("queue nothing yet") -- the mask already
    treats it as "not producible" (see
    ``test_production_respects_what_the_factory_says_it_can_produce``).
    """

    source = BRIDGE_SOURCE.read_text(encoding="utf-8", errors="replace")
    method = _method_body(source, "factoriesJson")
    assert "factoryCanProduceNow" in method, (
        "factoriesJson must gate the reported menu on the factory being finished; "
        "reporting a menu for an under-construction factory opens produce_unit "
        "early and the policy learns to avoid the action"
    )
    assert "producibleNamesFor(unit)" in method

    gate = _method_body(source, "factoryCanProduceNow")
    assert "isNonOperational" in gate, (
        "the engine's own operational flag must be consulted, not just progress"
    )
    assert "buildProgress" in gate, (
        "the progress check covers the window before the operational flag flips"
    )


@pytest.mark.skipif(not BRIDGE_SOURCE.is_file(), reason="bridge source not present")
def test_production_menu_is_only_reported_by_actual_factories() -> None:
    """A builder is not a factory: only ``isFactoryBuilding`` units get a menu."""

    source = BRIDGE_SOURCE.read_text(encoding="utf-8", errors="replace")
    method = _method_body(source, "factoriesJson")
    assert "isFactoryBuilding" in method
    assert "isBuilder" not in method, (
        "gating the factory list on isBuilder would omit factories and put "
        "production back behind the builder's build menu"
    )

def test_build_orders_use_the_broadcast_group_not_the_policy_group() -> None:
    """A build order must reach the builder, wherever its group happens to be.

    ``IronZeroBridge`` selects group members with ``unit.id % GROUP_COUNT``, so a
    map-placed builder belongs to exactly one numbered group for the whole
    episode -- but the policy picks ``group`` from a learned head with no idea
    where the builder is. On ``[p2]Lake (2p)`` the builder is unit 24 (group 0)
    and the command center is unit 23 (group 3); the action mask opens group 3
    because the command center lives there, and a build addressed to group 3 can
    never succeed. Measured over 200 updates on that map:
    ``BUILD_UNAVAILABLE=1620``, the largest rejection bucket by far.

    Broadcasting (``unit_group == cfg.groups``) orders every living friendly unit,
    which includes the builder wherever it sits and stays correct when a map
    places several builders or a factory produces more.
    """

    from ironzero.micro_config import DEFAULT_CONFIG
    from ironzero.rwpp.decoding import decode_action

    state = {
        "own_entities": [
            {"entity_id": 23, "hp": 4000.0, "max_hp": 4000.0, "x": 990.0, "y": 2040.0,
             "group_id": 3, "is_building": True, "is_factory": True},
            {"entity_id": 24, "hp": 170.0, "max_hp": 170.0, "x": 990.0, "y": 2070.0,
             "group_id": 0, "is_builder": True},
        ],
        "visible_enemy_entities": [],
        "global_state": {"funds": 4000.0},
        "resource_slots": [{"slot_id": index, "occupied": False} for index in range(6)],
    }
    broadcast = int(DEFAULT_CONFIG.groups)
    for kind in ("build_extractor", "build_structure"):
        for group in range(DEFAULT_CONFIG.groups + 1):
            action = np.array(
                [ACTION_KINDS.index(kind), group, 3, 3, 0], dtype=np.int64
            )
            decoded = decode_action(action, state, DEFAULT_CONFIG)
            assert decoded, f"{kind} produced no action for group {group}"
            assert decoded[0]["unit_group"] == broadcast, (
                f"{kind} sent unit_group={decoded[0]['unit_group']} for policy "
                f"group {group}; a numbered group may contain no builder, which is "
                "rejected as BUILD_UNAVAILABLE"
            )


def test_tactical_orders_still_honour_the_policy_group() -> None:
    """The broadcast remap must apply to *build* orders only.

    Move / attack / hold genuinely address a subset of the army, so collapsing
    them to broadcast would delete a real decision from the action space.
    """

    from ironzero.micro_config import DEFAULT_CONFIG
    from ironzero.rwpp.decoding import decode_action

    state = {
        "own_entities": [
            {"entity_id": 24, "hp": 170.0, "max_hp": 170.0, "x": 990.0, "y": 2070.0,
             "group_id": 0, "is_builder": True},
        ],
        "visible_enemy_entities": [],
        "global_state": {"funds": 4000.0},
        "resource_slots": [],
    }
    for kind in ("move_group", "attack_move", "hold", "stop"):
        for group in range(DEFAULT_CONFIG.groups):
            action = np.array(
                [ACTION_KINDS.index(kind), group, 3, 3, 0], dtype=np.int64
            )
            decoded = decode_action(action, state, DEFAULT_CONFIG)
            assert decoded[0]["unit_group"] == group, (
                f"{kind} must keep the policy's group ({group}), got "
                f"{decoded[0]['unit_group']}"
            )

def test_opening_extractor_curriculum_issues_one_real_build_order() -> None:
    """``reset()`` must queue an extractor through the normal bridge path.

    Why this exists: a from-scratch policy samples ``build_extractor`` about
    ``1/len(ACTION_KINDS)`` of the time and *every other order cancels the
    in-flight build* (that is why ``noop`` exists), so finishing a 140-tick
    structure by exploration is vanishingly unlikely. Measured on
    ``[p2]Lake (2p)``: ``income`` stayed at exactly 0.000 for 81 consecutive
    updates, and even after the mask bug was fixed it only moved on ~20% of
    episodes.

    The curriculum only *issues the order*. ``bonus_build`` still requires a
    finished building and ``income`` still pays from the observation stream, so no
    reward is handed out for free -- asserted below by checking the term is not
    touched here.
    """

    import inspect

    from ironzero.rwpp.config import RwppGameConfig
    from ironzero.rwpp.env import RwppDuelEnv

    # Opt-out must exist and default to on.
    assert RwppGameConfig().opening_extractor is True

    source = inspect.getsource(RwppDuelEnv._issue_opening_extractor)
    # Addressed via the broadcast group so it reaches the builder wherever it is.
    assert "micro.groups" in source, (
        "the opening order must broadcast, otherwise it misses the builder on any "
        "map whose builder is not in the sampled group"
    )
    # Sent through the shared step path, not a private socket write.
    assert "send_step" in source, (
        "the opening order must go through send_step so the game validates it "
        "exactly like a policy order"
    )
    # Only a free pool is a legal target.
    assert "occupied" in source
    # Must never break reset on a map that cannot honour it.
    assert "return None" in source

    reset_source = inspect.getsource(RwppDuelEnv.reset)
    assert "_issue_opening_extractor" in reset_source


def test_opening_curriculum_pays_no_reward_by_itself() -> None:
    """The curriculum must not be a reward source.

    ``bonus_build`` is counted from the observation stream by
    ``compute_step_shaping``, so issuing the order contributes nothing until the
    extractor finishes -- keeping the reward honest and policy-invariant.
    """

    from ironzero.rwpp.reward_shaping import compute_step_shaping

    source = compute_step_shaping.__doc__ or ""
    impl = __import__("inspect").getsource(compute_step_shaping)
    assert "bonus_build" in impl
    # The term is computed from own entity counts, not from receipts.
    assert "new_buildings" in impl

@pytest.mark.skipif(not BRIDGE_SOURCE.is_file(), reason="bridge source not present")
def test_terminal_done_message_carries_the_entity_lists() -> None:
    """``sendDone`` must include the terminal state, not just the verdict.

    It used to send only ``result``/``tick``/``terminated``. The last ``obs`` was
    therefore the last time the client saw any units, and
    ``merge_done_message`` -- which only merges keys that are present -- kept the
    *previous* frame. The closing decision was then scored against pre-terminal
    army counts:

    * ``losses`` (``previous_own_alive - own_alive``) scored 0 for the units that
      died on the final step;
    * ``potential`` used the pre-terminal hp, so the last shaping step was blind
      to the wipe-out it was being paid +/-1.0 for;
    * ``eval_report.json`` recorded the stale ``own_alive``/``enemy_alive``.

    Measured on a real ``IronZero5`` annihilation episode after the fix:
    ``own_alive=0``, ``enemy_alive=5``, ``losses=-1.25`` (= 5 x ``loss_penalty``),
    ``terminal=-1.0`` -- every casualty accounted for on the closing step.
    """

    source = BRIDGE_SOURCE.read_text(encoding="utf-8", errors="replace")
    done = _method_body(source, "sendDone")
    assert "own_entities" in done, (
        "done must carry own_entities, otherwise the terminal state is stale"
    )
    assert "visible_enemy_entities" in done, (
        "done must carry visible_enemy_entities; without it enemy_alive on the "
        "final decision is whatever the previous frame had"
    )
    assert "collectEntities" in done, (
        "the lists must come from the shared helper so they match observation"
    )

    shared = _method_body(source, "collectEntities")
    assert "isOwnUnit" in shared and "isEnemyUnit" in shared, (
        "team membership must use the engine-aware helpers, not a raw teamId "
        "comparison; neutral map objects must stay excluded from both lists"
    )


@pytest.mark.skipif(not BRIDGE_SOURCE.is_file(), reason="bridge source not present")
def test_done_message_still_delivers_a_verdict_when_the_snapshot_fails() -> None:
    """A failed entity snapshot must not swallow the result.

    The verdict is the load-bearing part of ``done``: the whole "wins were booked
    as timeouts" class of bugs came from losing it. Attaching the lists is an
    improvement, so it has to sit behind its own guard.
    """

    source = BRIDGE_SOURCE.read_text(encoding="utf-8", errors="replace")
    done = _method_body(source, "sendDone")
    assert "catch (RuntimeException" in done or "catch (Exception" in done, (
        "the entity snapshot needs its own try/catch; an exception there must not "
        "prevent writeLine of the verdict"
    )

def test_upgrade_order_is_not_issued_when_the_engine_says_nothing_upgrades() -> None:
    """A reported ``can_upgrade=false`` on every building must produce no order.

    The mask closes ``upgrade_building`` exactly when no own building reports
    ``can_upgrade``. ``decode_action`` used to fall back to "any own building" in
    that case, so a selected kind still produced a ``building_entity_id`` that the
    engine could only reject.

    Measured on ``[p2]Lake (2p)`` over 60 decisions: the mask was open on 47, yet
    every upgrade order was rejected; in 13 decisions the mask was *closed* while
    the decoder still emitted an order. In a real 25-update run this showed up as
    ``UPGRADE_UNAVAILABLE=647`` -- the largest rejection bucket, versus
    ``BUILD_UNAVAILABLE=3`` -- and it charged ``invalid_penalty`` every time.

    An older bridge that reports no ``can_upgrade`` field at all keeps the old
    fallback, so a missing capability is still survivable.
    """

    from ironzero.micro_config import DEFAULT_CONFIG
    from ironzero.rwpp.decoding import decode_action

    def state(units):
        return {
            "own_entities": units,
            "visible_enemy_entities": [],
            "global_state": {"funds": 4000.0},
            "resource_slots": [],
        }

    reported_not_upgradable = state([
        {"entity_id": 23, "hp": 4000.0, "max_hp": 4000.0, "x": 990.0, "y": 2040.0,
         "is_building": True, "can_upgrade": False},
        {"entity_id": 24, "hp": 170.0, "max_hp": 170.0, "x": 990.0, "y": 2070.0,
         "is_builder": True, "can_upgrade": False},
    ])
    action = np.array(
        [ACTION_KINDS.index("upgrade_building"), 4, 0, 0, 0], dtype=np.int64
    )
    assert decode_action(action, reported_not_upgradable, DEFAULT_CONFIG) == [], (
        "the engine said no upgrade is available; emitting an order guarantees "
        "UPGRADE_UNAVAILABLE and an invalid penalty"
    )

    # Older bridge: field absent -> keep the permissive fallback.
    unreported = state([
        {"entity_id": 23, "hp": 4000.0, "max_hp": 4000.0, "x": 990.0, "y": 2040.0,
         "is_building": True},
    ])
    decoded = decode_action(action, unreported, DEFAULT_CONFIG)
    assert decoded and decoded[0]["kind"] == "upgrade_building"

    # Something genuinely upgradable -> still ordered.
    upgradable = state([
        {"entity_id": 25, "hp": 600.0, "max_hp": 600.0, "x": 200.0, "y": 200.0,
         "is_building": True, "can_upgrade": True},
    ])
    decoded = decode_action(action, upgradable, DEFAULT_CONFIG)
    assert decoded and decoded[0]["building_entity_id"] == 25
