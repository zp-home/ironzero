# SPDX-License-Identifier: Apache-2.0
"""The engine decides the match; the tick cap is only a safety fuse.

This file replaces an earlier one whose central premise was wrong.

The old design inferred a winner on truncation from surviving-army ratios
(``_decide_by_army``). That heuristic had to be patched three times -- an
absolute hp-fraction gap, then a unit-count ratio, then a ``disconnected``
carve-out -- and still recorded a 2-vs-8 loss as a draw. Two measurements showed
the whole approach was the wrong tool:

1. **The engine already knows the result.** It reports ``victoryScreenShown`` /
   ``defeatScreenShown`` / ``wipedOut`` and the bridge forwards that as
   ``result``. The heuristic was second-guessing an authoritative signal.
2. **When the heuristic fired, it was describing a stalemate.** A 39000-tick
   Lake run ended with the enemy holding 22 units against our 1, and the engine
   reported *nothing* -- because the built-in AI never left its base. Calling
   that "defeat" was a guess dressed up as a verdict.

So the rule is now: engine verdict wins; anything else is a ``timeout``, and a
timeout costs ``timeout_penalty`` (because "ran out of time" must not be free).
The cap itself is derived from map size and player count
(``recommended_episode_ticks``) rather than being a fixed constant.
"""

from __future__ import annotations

import pytest

from ironzero.rwpp.config import (
    RwppGameConfig,
    recommended_episode_ticks,
)
from ironzero.rwpp.reward_shaping import compute_step_shaping


def _shaping(
    *,
    bridge_result: str | None = None,
    own_alive: int = 6,
    enemy_alive: int = 1,
    tick: int = 3600,
    disconnected: bool = False,
    done: bool = False,
    episode_ticks: int = 3600,
):
    game = RwppGameConfig(
        win_mode="native", units_per_team=8, episode_ticks=episode_ticks
    )
    micro = game.micro_config()
    state = {
        "tick": tick,
        "own_entities": [{"hp": 210.0, "max_hp": 210.0}] * own_alive,
        "visible_enemy_entities": [{"hp": 210.0, "max_hp": 210.0}] * enemy_alive,
        "global_state": {},
    }
    return compute_step_shaping(
        micro=micro,
        game=game,
        state=state,
        actions=[],
        receipts=[],
        message={"global_state": {}},
        bridge_result=bridge_result,
        potential=0.0,
        previous_potential=0.0,
        distance=0.5,
        previous_distance=0.5,
        own_alive=own_alive,
        enemy_alive=enemy_alive,
        previous_own_alive=own_alive,
        previous_enemy_alive=enemy_alive,
        strategy_component=0.0,
        disconnected=disconnected,
        done=done,
        own_max_ref=5250.0,
        enemy_max_ref=5250.0,
        episode_components={},
    )


# ---------------------------------------------------------------------------
# 1. The engine's verdict is used verbatim.
# ---------------------------------------------------------------------------
def test_a_native_victory_is_a_terminated_win() -> None:
    shaping = _shaping(bridge_result="victory")
    assert shaping.terminated
    assert not shaping.truncated
    assert shaping.last_result == "victory"
    assert shaping.components["terminal"] == pytest.approx(1.0)


def test_a_native_defeat_is_a_terminated_loss() -> None:
    shaping = _shaping(bridge_result="defeat")
    assert shaping.terminated
    assert not shaping.truncated
    assert shaping.last_result == "defeat"
    assert shaping.components["terminal"] == pytest.approx(-1.0)


def test_the_engine_verdict_beats_the_army_counts() -> None:
    """A declared victory is a victory even if we have fewer units left.

    The engine knows about objectives, buildings and team elimination; a raw unit
    count does not. This is the case the old heuristic got wrong.
    """

    shaping = _shaping(bridge_result="victory", own_alive=1, enemy_alive=6)
    assert shaping.last_result == "victory"
    assert shaping.components["terminal"] == pytest.approx(1.0)


def test_the_engine_verdict_is_not_awarded_a_win_bonus_on_defeat() -> None:
    shaping = _shaping(bridge_result="defeat", own_alive=6, enemy_alive=1)
    assert shaping.last_result == "defeat"
    assert shaping.components["terminal"] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# 2. No engine verdict => timeout, and a timeout costs something.
# ---------------------------------------------------------------------------
def test_reaching_the_tick_cap_without_a_verdict_is_a_timeout() -> None:
    shaping = _shaping(tick=3600, episode_ticks=3600)
    assert shaping.truncated
    assert not shaping.terminated
    assert shaping.last_result == "timeout"


def test_a_timeout_pays_the_configured_penalty() -> None:
    shaping = _shaping(tick=3600, episode_ticks=3600)
    assert shaping.components["timeout"] == pytest.approx(
        -RwppGameConfig().timeout_penalty
    )
    assert shaping.components["timeout"] < 0.0


def test_a_timeout_pays_nothing_into_terminal() -> None:
    """Only the *engine* may move ``terminal``. No verdict, no terminal value."""

    shaping = _shaping(tick=3600, episode_ticks=3600)
    assert shaping.components["terminal"] == 0.0


@pytest.mark.parametrize(
    "own_alive,enemy_alive",
    [
        (6, 1),   # looks like a clear win  -> still a timeout now
        (1, 6),   # looks like a clear loss -> still a timeout now
        (2, 8),   # the measured 60-update run
        (13, 13),  # even
        (25, 0),  # enemy wiped but no verdict (legacy bridge path handles it)
    ],
)
def test_army_counts_alone_never_decide_a_timeout(
    own_alive: int, enemy_alive: int
) -> None:
    """The ratio heuristic is gone: unit counts no longer imply a winner."""

    shaping = _shaping(
        own_alive=own_alive, enemy_alive=enemy_alive, tick=3600, episode_ticks=3600
    )
    if own_alive > 0 and enemy_alive > 0:
        assert shaping.last_result == "timeout"
        assert shaping.components["terminal"] == 0.0


def test_a_stalemate_pays_less_than_resolving_the_match() -> None:
    """Sitting still for the whole cap must cost more than a decided game.

    The assertion is on the *timeout* term specifically, not on the total reward:
    a neutral step can still earn dense shaping (``bonus_build`` for a structure
    that finished, ``potential`` for ground gained), and conflating the two would
    make this test depend on unrelated reward terms.
    """

    stalling = _shaping(tick=3600, episode_ticks=3600)
    decided = _shaping(bridge_result="victory")
    assert stalling.components["timeout"] < 0.0
    assert decided.components["timeout"] == 0.0
    assert stalling.components["timeout"] < decided.components["terminal"]


# ---------------------------------------------------------------------------
# 3. Disconnects are truncations too, and never invent a verdict.
# ---------------------------------------------------------------------------
def test_a_disconnect_never_awards_a_win() -> None:
    shaping = _shaping(disconnected=True, done=True, own_alive=6, enemy_alive=1)
    assert shaping.last_result == "timeout"
    assert shaping.components["terminal"] == 0.0


def test_a_disconnect_never_awards_a_defeat_either() -> None:
    """A dropped stream is missing data, not evidence.

    The previous revision carved out an asymmetry ("defeat is allowed, victory is
    not") to rescue obvious losses. With the engine as the source of truth that
    carve-out is unnecessary: if we really lost, the engine says so *before* the
    socket dies. A disconnect with no verdict is simply a timeout.
    """

    shaping = _shaping(disconnected=True, done=False, own_alive=2, enemy_alive=8)
    assert shaping.last_result == "timeout"
    assert shaping.components["terminal"] == 0.0


# ---------------------------------------------------------------------------
# 4. The cap is sized from geometry, not hard-coded.
# ---------------------------------------------------------------------------
def test_recommended_cap_scales_with_map_size() -> None:
    small = recommended_episode_ticks(1000.0, 1000.0, 2)
    large = recommended_episode_ticks(4000.0, 4000.0, 2)
    assert large > small


def test_recommended_cap_scales_with_player_count() -> None:
    two = recommended_episode_ticks(2600.0, 2600.0, 2)
    four = recommended_episode_ticks(2600.0, 2600.0, 4)
    assert four > two
    # ...but sub-linearly: overlapping pushes mean 4 players do not need 3x.
    assert four < 2 * two


def test_recommended_cap_is_inside_the_accepted_range() -> None:
    for width, height in [(200.0, 200.0), (2600.0, 2600.0), (20000.0, 20000.0)]:
        ticks = recommended_episode_ticks(width, height, 2)
        assert 120 <= ticks <= 216000


def test_recommended_cap_is_long_enough_for_a_real_match() -> None:
    """A Lake-sized map must allow far more than one approach.

    Measured reference: a 39000-tick Lake episode never resolved because the AI
    never attacked. The cap must not be the thing that ends a game the engine
    could have decided, so it has to be generous for the map being trained.
    """

    tick_cap = recommended_episode_ticks(2600.0, 2600.0, 2)
    assert tick_cap >= 15000


def test_zero_episode_ticks_derives_the_cap() -> None:
    game = RwppGameConfig(map_width_px=2600.0, map_height_px=2600.0)
    assert game.episode_ticks == 0
    assert game.resolved_episode_ticks() == recommended_episode_ticks(
        2600.0, 2600.0, game.num_teams
    )


def test_explicit_episode_ticks_is_respected() -> None:
    game = RwppGameConfig(episode_ticks=4321)
    assert game.resolved_episode_ticks() == 4321


def test_micro_config_carries_the_resolved_cap() -> None:
    game = RwppGameConfig(
        map_width_px=2600.0, map_height_px=2600.0, units_per_team=8
    )
    assert game.micro_config().episode_ticks == game.resolved_episode_ticks()


def test_timeout_penalty_is_smaller_than_a_lost_game() -> None:
    """Stalling must not be *preferable* to losing outright.

    If a timeout cost more than a defeat, a policy near the cap would be pushed to
    lose on purpose rather than risk running out of time.
    """

    game = RwppGameConfig()
    micro = game.micro_config()
    assert game.timeout_penalty < abs(micro.loss_reward)
