# SPDX-License-Identifier: Apache-2.0
"""Truncation adjudication: a clearly-won tick-capped battle must be a win.

Regression guard for a real measurement: a probe episode ended with 4 own units
alive against 1 enemy and was recorded as ``timeout`` with ``terminal = 0.0``.
The old rule required an absolute HP-fraction gap of 0.08, but the fractions are
normalised by the *starting* army totals, so in a long battle both sides shrink
towards zero (0.046 vs 0.040 here, a gap of 0.006) and the threshold became
unreachable exactly when the battle was decided. Every such episode paid the
policy nothing for winning, which is indistinguishable from losing.

The fix decides on surviving units first (annihilation is a win) and only falls
back to a relative-strength comparison for equal-sized armies.
"""

from __future__ import annotations

import pytest

from ironzero.rwpp.reward_shaping import (
    TIMEOUT_TERMINAL_SCALE,
    _decide_by_army,
    compute_step_shaping,
)
from ironzero.rwpp.config import RwppGameConfig


@pytest.mark.parametrize(
    "own_alive,enemy_alive,own_frac,enemy_frac,expected",
    [
        # Annihilation is decisive regardless of hp.
        (25, 0, 1.00, 0.00, "victory"),
        (0, 25, 0.00, 1.00, "defeat"),
        (0, 0, 0.0, 0.0, "timeout"),
        # The measured regression: 4 vs 1, both armies badly damaged. The old
        # rule read the hp gap (0.046 - 0.040 = 0.006) against an absolute 0.08
        # and called a 4-vs-1 win a timeout.
        (4, 1, 0.046, 0.040, "victory"),
        (4, 1, 0.160, 0.040, "victory"),
        # A materially larger surviving army wins...
        (8, 6, 0.320, 0.240, "victory"),
        (2, 1, 0.080, 0.040, "victory"),
        (24, 18, 0.80, 0.70, "victory"),
        # ...and a materially smaller one loses.
        (4, 14, 0.160, 0.560, "defeat"),
        (18, 24, 0.70, 0.80, "defeat"),
        # Armies of comparable size are a draw even if hp differs: nothing was
        # decided. This is the case a naive "more units wins" rule got wrong
        # (21 vs 23 was scored as a defeat).
        (13, 13, 0.520, 0.520, "timeout"),
        (19, 19, 0.300, 0.250, "timeout"),
        (21, 23, 0.40, 0.55, "timeout"),
        (23, 21, 0.55, 0.40, "timeout"),
        # Half the units is decisive even when the survivors are healthier: the
        # rule is about surviving *army size*, not about who has more hp left.
        (1, 2, 0.200, 0.100, "defeat"),
        (2, 1, 0.100, 0.200, "victory"),
    ],
)
def test_battle_adjudication(
    own_alive: int,
    enemy_alive: int,
    own_frac: float,
    enemy_frac: float,
    expected: str,
) -> None:
    assert (
        _decide_by_army(
            own_alive=own_alive,
            enemy_alive=enemy_alive,
            own_frac=own_frac,
            enemy_frac=enemy_frac,
        )
        == expected
    )


def _truncated(own_hps: list[float], enemy_hps: list[float]):
    """Run one truncating shaping step for the given surviving armies."""

    game = RwppGameConfig(win_mode="annihilation", units_per_team=25, episode_ticks=3600)
    micro = game.micro_config()
    own = [{"hp": hp, "max_hp": 210.0} for hp in own_hps]
    enemies = [{"hp": hp, "max_hp": 210.0} for hp in enemy_hps]
    state = {
        "tick": 3600,
        "own_entities": own,
        "visible_enemy_entities": enemies,
        "global_state": {},
    }
    return compute_step_shaping(
        micro=micro,
        game=game,
        state=state,
        actions=[],
        receipts=[],
        message={"global_state": {}},
        bridge_result=None,
        potential=0.0,
        previous_potential=0.0,
        distance=0.5,
        previous_distance=0.5,
        own_alive=len(own),
        enemy_alive=len(enemies),
        previous_own_alive=len(own),
        previous_enemy_alive=len(enemies),
        strategy_component=0.0,
        disconnected=False,
        done=True,
        own_max_ref=5250.0,
        enemy_max_ref=5250.0,
        episode_components={},
    )


def test_a_clearly_won_truncated_battle_pays_the_win() -> None:
    """The exact measured case: 4 badly-damaged units left vs 1."""

    shaping = _truncated([60.0] * 4, [210.0])
    assert shaping.last_result == "victory"
    assert shaping.components["terminal"] == pytest.approx(
        1.0 * TIMEOUT_TERMINAL_SCALE
    )
    assert shaping.truncated is True


def test_a_clearly_lost_truncated_battle_pays_the_loss() -> None:
    shaping = _truncated([60.0] * 2, [210.0] * 12)
    assert shaping.last_result == "defeat"
    assert shaping.components["terminal"] == pytest.approx(
        -1.0 * TIMEOUT_TERMINAL_SCALE
    )


def test_an_even_truncated_battle_pays_nothing() -> None:
    shaping = _truncated([210.0] * 5, [210.0] * 5)
    assert shaping.last_result == "timeout"
    assert shaping.components["terminal"] == 0.0


def test_a_disconnect_never_awards_a_win() -> None:
    """A dead connection must not be adjudicated as a victory."""

    game = RwppGameConfig(win_mode="annihilation", units_per_team=25, episode_ticks=3600)
    micro = game.micro_config()
    state = {
        "tick": 3600,
        "own_entities": [{"hp": 210.0, "max_hp": 210.0}] * 6,
        "visible_enemy_entities": [{"hp": 210.0, "max_hp": 210.0}],
        "global_state": {},
    }
    shaping = compute_step_shaping(
        micro=micro,
        game=game,
        state=state,
        actions=[],
        receipts=[],
        message={"global_state": {}},
        bridge_result=None,
        potential=0.0,
        previous_potential=0.0,
        distance=0.5,
        previous_distance=0.5,
        own_alive=6,
        enemy_alive=1,
        previous_own_alive=6,
        previous_enemy_alive=1,
        strategy_component=0.0,
        disconnected=True,
        done=True,
        own_max_ref=5250.0,
        enemy_max_ref=5250.0,
        episode_components={},
    )
    assert shaping.last_result == "timeout"
    assert shaping.components["terminal"] == 0.0


def test_a_disconnect_does_award_a_defeat() -> None:
    """But a wipe-out is unambiguous even when the stream stops.

    The asymmetry is deliberate. Our own units are always fully visible (they are
    ours), so seeing them gone is real evidence -- whereas "our army looks bigger
    right now" is not evidence that the enemy is beaten (they may just be out of
    sight). Awarding defeat on missing data is honest; awarding victory is a false
    positive that corrupts the win rate.

    Why this matters on the real game: a 60-update run reported every eval episode
    as ``result=timeout`` with ``own_alive=2, enemy_alive=8`` at ``ticks=5351``
    against a 6000 cap -- the bridge had stopped before the cap, and obvious
    defeats were being paid ``terminal=0.0``.
    """

    game = RwppGameConfig(win_mode="native", units_per_team=8, episode_ticks=6000,
                          map_relative="maps/skirmish/[p2]Lake (2p).tmx")
    micro = game.micro_config()
    state = {
        "tick": 5351,
        "own_entities": [{"hp": 4000.0, "max_hp": 4000.0}] * 2,
        "visible_enemy_entities": [{"hp": 210.0, "max_hp": 210.0}] * 8,
        "global_state": {},
    }
    shaping = compute_step_shaping(
        micro=micro,
        game=game,
        state=state,
        actions=[],
        receipts=[],
        message={"global_state": {}},
        bridge_result=None,
        potential=0.0,
        previous_potential=0.0,
        distance=0.5,
        previous_distance=0.5,
        own_alive=2,
        enemy_alive=8,
        previous_own_alive=2,
        previous_enemy_alive=8,
        strategy_component=0.0,
        disconnected=True,
        done=False,
        own_max_ref=4170.0,
        enemy_max_ref=4170.0,
        episode_components={},
    )
    assert shaping.truncated
    assert shaping.last_result == "defeat"
    assert shaping.components["terminal"] == pytest.approx(
        -1.0 * TIMEOUT_TERMINAL_SCALE
    )
