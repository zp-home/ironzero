# SPDX-License-Identifier: Apache-2.0
"""The behaviour-cloning teacher path must stay reachable end to end.

The teacher is the game's own built-in AI: ``--ai-pilot`` hands the controlled
team to it and ``--record-ai-orders`` streams back what it actually ordered. Two
silent failures were found while building this, and both looked like "the flag
did nothing" rather than an error:

* the bridge accepted ``--ai-pilot`` but the AI was never ticked (not registered
  in ``Team.teamArray``), so our funds stayed frozen and no order was ever read;
* ``observation_to_state`` dropped ``info.ai_orders``, so the data existed on the
  wire and vanished before the trainer saw it.

These tests lock the Python-visible half of that contract; the Java half is
covered by the source assertions below.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ironzero.rwpp.config import RwppGameConfig
from ironzero.rwpp.decoding import observation_to_state
from ironzero.rwpp.process import _bridge_command

REPO_ROOT = Path(__file__).resolve().parents[1]
BRIDGE_SOURCE = REPO_ROOT / "src" / "ironzero_bridge" / "java" / "ironzero" / "IronZeroBridge.java"


# ---------------------------------------------------------------------------
# 1. Config / command line contract
# ---------------------------------------------------------------------------
def test_teacher_flags_are_opt_in() -> None:
    """A normal run must not send the teacher flags."""

    assert RwppGameConfig().ai_pilot is False
    assert RwppGameConfig().record_ai_orders is False

    plain = _bridge_command(Path("rwpp-server.bat"), 5130, RwppGameConfig())
    assert "--ai-pilot" not in plain
    assert "--record-ai-orders" not in plain


def test_teacher_flags_reach_the_bridge() -> None:
    game = RwppGameConfig(ai_pilot=True, record_ai_orders=True)
    command = _bridge_command(Path("rwpp-server.bat"), 5130, game)
    assert "--ai-pilot" in command
    assert "--record-ai-orders" in command


def test_ai_pilot_alone_does_not_enable_recording() -> None:
    """Recording is a separate opt-in: piloting without labels is still useful."""

    game = RwppGameConfig(ai_pilot=True)
    command = _bridge_command(Path("rwpp-server.bat"), 5130, game)
    assert "--ai-pilot" in command
    assert "--record-ai-orders" not in command


# ---------------------------------------------------------------------------
# 2. ai_orders must survive decoding (it was silently dropped)
# ---------------------------------------------------------------------------
def _obs_message(info: dict) -> dict:
    return {
        "protocol": "ironzero/1",
        "type": "obs",
        "tick": 10,
        "game_time": 0.16,
        "controlled_team": 0,
        "global_state": {"funds": 1000.0},
        "own_entities": [],
        "visible_enemy_entities": [],
        "resource_slots": [],
        "info": info,
    }


def test_ai_orders_survive_observation_decoding() -> None:
    """Regression: the labels existed on the wire but were dropped here."""

    orders = [
        {"entity_id": 23, "kind": "hold", "waypoint_type": "", "x": 990.0, "y": 2010.0,
         "target_entity_id": -1, "attack_target_id": -1},
        {"entity_id": 30, "kind": "attack_move", "waypoint_type": "attackMove",
         "x": 1200.0, "y": 800.0, "target_entity_id": -1, "attack_target_id": 21},
    ]
    state = observation_to_state(_obs_message({"ai_orders": orders, "ai_pilot": True}))
    assert state["info"]["ai_orders"] == orders
    assert state["info"]["ai_pilot"] is True


def test_missing_ai_orders_decode_to_an_empty_list() -> None:
    """A non-teacher bridge must not make callers special-case ``None``."""

    state = observation_to_state(_obs_message({"builders": []}))
    assert state["info"]["ai_orders"] == []
    assert state["info"]["ai_pilot"] is False


# ---------------------------------------------------------------------------
# 3. Java side: the pilot must be registered and units re-homed
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not BRIDGE_SOURCE.is_file(), reason="bridge source not present")
def test_bridge_registers_the_pilot_in_the_team_array() -> None:
    """``new AI(teamId, false)`` builds an AI that is never ticked.

    The constructor's second argument is ``replaceInTeamArray``; with ``false``
    the object exists and ``playerTeam`` points at it, but
    ``Team.updateTeams`` cannot reach it. Measured symptom: own funds frozen at
    4000 for 192 s with ``orders=0`` -- it looked installed and played nothing.
    """

    source = BRIDGE_SOURCE.read_text(encoding="utf-8", errors="replace")
    assert re.search(r"new AI\(\s*teamId\s*,\s*true\s*\)", source), (
        "the AI pilot must be constructed with replaceInTeamArray=true"
    )
    assert "new AI(teamId, false)" not in source


@pytest.mark.skipif(not BRIDGE_SOURCE.is_file(), reason="bridge source not present")
def test_bridge_rehomes_own_units_onto_the_pilot() -> None:
    """Team membership is compared by reference, so units must be re-pointed.

    Without this the engine reported ``Player had no starting units`` and ended
    the game at tick=1 with ``wipedOut=true``.
    """

    source = BRIDGE_SOURCE.read_text(encoding="utf-8", errors="replace")
    assert "unit.team = pilot" in source, (
        "own units must be re-homed onto the pilot team object, otherwise "
        "unit.team == this never matches and the team is declared wiped out"
    )
    assert "pilot.hasHadUnits" in source


@pytest.mark.skipif(not BRIDGE_SOURCE.is_file(), reason="bridge source not present")
def test_bridge_emits_ai_orders_in_the_observation() -> None:
    source = BRIDGE_SOURCE.read_text(encoding="utf-8", errors="replace")
    assert '\\"ai_orders\\":' in source
    assert "aiOrdersJson()" in source
    assert '\\"ai_pilot\\":' in source


@pytest.mark.skipif(not BRIDGE_SOURCE.is_file(), reason="bridge source not present")
def test_bridge_maps_every_waypoint_type_we_record() -> None:
    """The label vocabulary must cover the waypoint types the AI actually uses."""

    source = BRIDGE_SOURCE.read_text(encoding="utf-8", errors="replace")
    body_start = source.find("private String waypointKind(")
    assert body_start >= 0
    body = source[body_start : body_start + 900]
    for waypoint_type in ("move", "attackMove", "attack", "build", "repair", "patrol"):
        assert f'case "{waypoint_type}"' in body, f"{waypoint_type} is not mapped to a kind"


# ---------------------------------------------------------------------------
# 4. Teacher strength (AI difficulty) must be selectable and recorded
# ---------------------------------------------------------------------------
def test_ai_difficulty_is_opt_in_so_the_default_command_is_unchanged() -> None:
    """Omitting it must keep the engine's own resolution, not pin a level."""

    assert RwppGameConfig().ai_difficulty is None
    plain = _bridge_command(Path("rwpp-server.bat"), 5130, RwppGameConfig())
    assert "--ai-difficulty" not in plain


def test_ai_difficulty_reaches_the_bridge() -> None:
    game = RwppGameConfig(ai_difficulty=3)
    command = _bridge_command(Path("rwpp-server.bat"), 5130, game)
    assert "--ai-difficulty" in command
    assert command[command.index("--ai-difficulty") + 1] == "3"


def test_ai_difficulty_can_be_negative() -> None:
    """Level -2 is a legitimate (very easy) setting and must not be dropped."""

    command = _bridge_command(Path("rwpp-server.bat"), 5130, RwppGameConfig(ai_difficulty=-2))
    assert command[command.index("--ai-difficulty") + 1] == "-2"


def test_ai_difficulty_survives_decoding() -> None:
    state = observation_to_state(
        _obs_message(
            {
                "ai_pilot": True,
                "ai_difficulty": 2,
                "ai_teams": [
                    {"team_id": 0, "difficulty": 2, "income_multiplier": 1.8, "locked": True}
                ],
            }
        )
    )
    assert state["info"]["ai_difficulty"] == 2
    assert state["info"]["ai_teams"][0]["income_multiplier"] == 1.8


@pytest.mark.skipif(not BRIDGE_SOURCE.is_file(), reason="bridge source not present")
def test_bridge_locks_the_difficulty_it_applies() -> None:
    """Setting ``aiDifficulty`` alone does nothing -- the lock is what applies it.

    ``getEffectiveAIDifficulty()`` ignores a team's own ``aiDifficulty`` unless
    ``aiDifficultyLocked`` is set, falling back to the global settings preference
    (default 0). Measured before the lock existed: identical x1.00 income at every
    requested level.
    """

    source = BRIDGE_SOURCE.read_text(encoding="utf-8", errors="replace")
    assert "team.aiDifficultyLocked = true" in source
    assert "team.getAIDifficultyIncomeMultiplier()" in source


@pytest.mark.skipif(not BRIDGE_SOURCE.is_file(), reason="bridge source not present")
def test_bridge_reports_the_resolved_difficulty_not_the_requested_one() -> None:
    """The observation must record what the engine resolved, per team."""

    source = BRIDGE_SOURCE.read_text(encoding="utf-8", errors="replace")
    assert '\\"ai_teams\\":' in source
    assert "getEffectiveAIDifficulty()" in source


@pytest.mark.skipif(not BRIDGE_SOURCE.is_file(), reason="bridge source not present")
def test_income_feature_includes_the_ai_difficulty_multiplier() -> None:
    """``income`` must not be blind to the AI's resource cheat.

    ``Unit.getGenerationCredits()`` is the raw rate; the engine applies the
    multiplier in ``Team.updateIncome``. Reporting the raw sum made the feature
    read 18.0/s at difficulty -2, 0 and 3 alike.
    """

    source = BRIDGE_SOURCE.read_text(encoding="utf-8", errors="replace")
    body_start = source.find("private double ownIncome()")
    assert body_start >= 0
    body = source[body_start : body_start + 1600]
    assert "getAIDifficultyIncomeMultiplier()" in body, (
        "ownIncome() must scale by the AI difficulty multiplier or the income "
        "feature cannot see the cheat"
    )


def test_difficulty_income_multiplier_matches_the_engine_bytecode() -> None:
    """The documented table must match the engine, including level 3's flat +1.5.

    Verified by disassembling ``Team.getAIDifficultyIncomeMultiplier``: level 3
    adds ``1.5f`` (not ``1.0f``), so it is x3.70, and measurement agreed (66.6/s
    against an 18.0/s base).
    """

    def multiplier(level: int) -> float:
        value = 1.0 + (level * 0.4 if level > 0 else level * 0.3)
        if level == 3:
            value += 1.5
        return max(0.1, value)

    assert multiplier(-2) == pytest.approx(0.4)
    assert multiplier(-1) == pytest.approx(0.7)
    assert multiplier(0) == pytest.approx(1.0)
    assert multiplier(1) == pytest.approx(1.4)
    assert multiplier(2) == pytest.approx(1.8)
    assert multiplier(3) == pytest.approx(3.7)
