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

import pytest

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
