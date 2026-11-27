# SPDX-License-Identifier: Apache-2.0
"""Single real-game environment driven through the IronZero bridge.

The game itself is not resettable in-process, so ``reset()`` starts a fresh
headless process. That costs a few seconds of JVM startup, which is why
:class:`ironzero.training.vec_rwpp.VecRwppEnv` resets several instances in
parallel and keeps them running across PPO rollouts.

This module only orchestrates: process launch/pacing lives in
:mod:`ironzero.rwpp.process`, socket framing in :mod:`ironzero.rwpp.transport`,
message/action conversion in :mod:`ironzero.rwpp.decoding`, per-decision
plumbing (sending the step, absorbing the reply, packing ``info``) in
:mod:`ironzero.rwpp.stepping`, fog of war in :mod:`ironzero.rwpp.vision`,
potential terms in :mod:`ironzero.rwpp.reward` and the reward decomposition in
:mod:`ironzero.rwpp.reward_shaping`.
"""

from __future__ import annotations

import socket
from typing import Any, Callable

import numpy as np

from ..micro_config import MicroDuelConfig
from ..mock_engine import ACTION_KINDS
from ..observations import build_masks_for_config as build_masks, encode_observation
from ..observation.menus import (
    build_menu,
    is_defence_type,
    menu_index,
    produce_options,
    upgradable_buildings,
)
from .config import RwppGameConfig
from .decoding import decode_action, observation_to_state
from .errors import RwppEnvError
from .macro import ResolvedMacro, economy_summary, resolve_macro
from .process import _ProcLike, launch_rwpp_process
from .reward import army_distance_norm, army_hp_reference, combat_value_potential
from .reward_shaping import combat_army_distance_norm, compute_step_shaping
from .session import await_observation, drain_verdict, shutdown_game, start_game
from .spaces import _BoxSpace, _MultiDiscreteSpace
from .stepping import absorb_message, build_step_info, send_step
from .macro import resolve_macro
from .strategy import refresh_strategy, strategy_potential, strategy_summary
from .transport import _LineReader
from .vision import apply_vision

#: Consecutive decisions with no forward progress before a curriculum build order
#: is re-issued.
#:
#: Repeating a build order for the *same* target is **not** a no-op in this engine:
#: it restarts the construction, pinning ``build_progress`` at exactly 0.0 while
#: every receipt still reports ``accepted: true``. Measured A/B on
#: ``[p2]Lake (2p)`` with a policy that issues nothing
#: (``.runtime/ab_reissue.py``): ordering once reached a finished extractor at
#: decision 140, ordering on every decision never finished it in 1200 decisions.
#:
#: The curriculum therefore stays silent while progress advances and only re-orders
#: after ~2 s (20 decisions x 10 ticks) of no movement, which is long enough not to
#: trip on a single slow tick and short enough to recover from a policy that pulled
#: the builder away.
STALL_DECISIONS = 20


def _building_tier_total(units: Any) -> int:
    """Tiers *gained above the starting tier*, counting buildings only.

    This is the quantity ``bonus_upgrade`` pays for, so what it must not include
    matters more than what it includes:

    * **not units.** Every unit carries ``tech_level`` 1, so a per-entity sum makes
      producing a scout look exactly like upgrading an extractor. Measured: with the
      old per-entity sum, ``bonus_upgrade`` fired on 472 of 498 decisions while no
      building ever left T1 -- the policy was paid the upgrade bonus for massing
      scouts, and it did exactly that for 15 straight updates
      (``.runtime/tier_reward_audit.json``).
    * **not the starting tier.** A building begins at T1, so counting ``tech_level``
      outright pays the bonus once for the command centre merely existing. Only the
      tiers above 1 are achievements.

    Returns 0 for a payload that reports no ``tech_level`` at all, which keeps the
    reward bit-identical on the mock engine and on older bridges -- the same
    "missing field reads as zero" contract the surrounding counters use.
    """

    total = 0
    for unit in units or []:
        if not isinstance(unit, dict) or not unit.get("is_building"):
            continue
        try:
            level = int(float(unit.get("tech_level") or 0))
        except (TypeError, ValueError):
            continue
        if level > 1:
            total += level - 1
    return total


def _is_operational(unit: dict[str, Any]) -> bool:
    """Whether the engine considers this structure finished and usable.

    ``is_operational`` is the engine's own answer (``!isNonOperational()``, i.e.
    ``isVirtualUnit() || buildProgress < 1.0 || team == disconnectedTeam``), so it
    is preferred over ``build_progress`` wherever the question is "may this
    structure act yet?".

    The two signals genuinely disagree: a T1 extractor that is finished can be
    upgraded to T2/T3, and during the upgrade ``build_progress`` drops below 1.0
    again while ``is_operational`` stays true because the building still produces.
    A curriculum that gates on raw progress therefore treats a policy's upgrade
    order as "the extractor is not finished" and refuses to advance -- measured on
    a masked-random Lake episode: ``extractorT3`` at ``income`` 1.0 for 2006
    decisions with the factory stage never reached.

    Falls back to ``build_progress`` for older bridge builds that do not emit the
    field, which is also what the tests' synthetic entities carry.
    """

    if "is_operational" in unit:
        # A structure that has ever been operational is treated as operational:
        # a mid-upgrade building reports ``is_operational: false`` for the duration
        # of the upgrade, and the curriculum must not mistake that for "never
        # built". This is safe because the only consequence is that an *earlier*
        # stage stops re-issuing its order -- the later stages still wait for their
        # own conditions, and ``bonus_build``/``income`` are paid from the
        # observation stream regardless.
        return bool(unit.get("is_operational")) or float(
            unit.get("build_progress", 0.0) or 0.0
        ) >= 1.0
    return float(unit.get("build_progress", 1.0) or 0.0) >= 1.0



#: Rough combat value per unit type. Higher is better; unknown types rank 0.
#:
#: These are an *ordering*, not a balance claim. What matters is that ``scout`` sits
#: below every real combat unit, because a menu-driven "first entry" policy was
#: measured to build nothing but scouts while three land factories stood idle
#: (endgame probe on Lake at difficulty 1).
_UNIT_COMBAT_RANK: dict[str, int] = {
    "scout": 1,
    "hoverTank": 3,
    "tank": 4,
    "c_tank": 5,
    "artillery": 5,
    "c_artillery": 6,
    "lightGunship": 4,
    "c_helicopter": 5,
    "gunship": 5,
    "bomber": 6,
    "c_bomber": 6,
}

#: Rough price per unit type, in credits. Used only to compare value per credit.
_UNIT_PRICE: dict[str, int] = {
    "scout": 700,
    "hoverTank": 450,
    "tank": 350,
    "c_tank": 500,
    "artillery": 900,
    "c_artillery": 1000,
    "lightGunship": 700,
    "c_helicopter": 900,
    "gunship": 900,
    "bomber": 1000,
    "c_bomber": 1200,
}


def _unit_combat_rank(name: str) -> int:
    """Combat value of a producible unit name; 0 when unknown."""
    return _UNIT_COMBAT_RANK.get(name.split(".")[-1], 0)


def _unit_value_per_credit(name: str) -> float:
    """Combat value per credit, so an affordable unit beats a better unaffordable one.

    Ordering by absolute strength alone was measured to *empty* the army: the
    preferred unit on a land factory is ``artillery`` at 900 credits, the factory
    waited for a bank it never saw, and the episode ended with zero mobile units
    while income sat at 42 and three factories were idle. A 350-credit tank that is
    actually built is worth more than a 900-credit artillery that is not.
    """
    key = name.split(".")[-1]
    rank = _UNIT_COMBAT_RANK.get(key, 0)
    if rank <= 0:
        return 0.0
    price = max(1, _UNIT_PRICE.get(key, 500))
    return rank / float(price)


def _combat_factories(state: dict) -> list[dict]:
    """Factories that can produce a *real* combat unit.

    A factory qualifies only when its menu contains something better than a scout,
    and the command centre never qualifies. That building's purpose is to make
    builders; its ``scout`` entry is the weakest unit on the map, so counting it as
    a combat factory both mislabels the base and -- because
    ``produce_options`` lists it first -- captures every production order.

    Measured consequence of the old predicate on Lake at difficulty 1: three
    ``landFactory`` buildings stood idle for a whole episode while the army was six
    ``scout`` against an enemy holding tanks, artillery and gunships.
    """
    out: list[dict] = []
    for entry in state.get("factories") or []:
        if not isinstance(entry, dict):
            continue
        type_name = str(entry.get("unit_type") or "").lower()
        if "commandcenter" in type_name or "command_centre" in type_name:
            continue
        menu = [str(name) for name in (entry.get("can_produce") or [])]
        if any(_unit_combat_rank(name) > _unit_combat_rank("scout") for name in menu):
            out.append(entry)
    return out


def _occupied_extractors(state: dict) -> int:
    """How many resource pools currently hold an extractor.

    Read from the bridge's occupancy view rather than by counting ``extractor``
    entities: a pool flips ``occupied`` only when the hull is streamed, which is
    the same signal the expansion stage latches on, so the two stages agree on
    whether capacity remains.
    """

    return sum(
        1
        for slot in state.get("resource_slots") or []
        if isinstance(slot, dict) and slot.get("occupied")
    )


class RwppDuelEnv:
    """Single real-game environment driven through the IronZero bridge."""

    metadata = {"render_modes": [], "name": "IronZeroRwppDuel-v0"}

    def __init__(
        self,
        game: RwppGameConfig | None = None,
        *,
        port: int = 5140,
        seed: int = 0,
        micro_config: MicroDuelConfig | None = None,
        launcher: Callable[[RwppGameConfig, int], _ProcLike] | None = None,
        episode_prefix: str = "rwpp",
        strategy: Any | None = None,
    ) -> None:
        self.game = game or RwppGameConfig()
        self.port = int(port)
        self.seed = int(seed)
        self.micro = micro_config or self.game.micro_config()
        self._launcher = launcher or launch_rwpp_process
        self.episode_prefix = episode_prefix
        self.strategy = strategy
        self.directive: Any | None = None
        self.last_strategy_potential: float | None = None
        # The commander's economy plan, resolved against this deployment's
        # configuration. Always present, so every build stage can read a budget
        # without repeating the "is there a directive" check. It starts as the
        # configured default, i.e. exactly the pre-macro behaviour, and is
        # refreshed wherever the directive is refreshed.
        self.macro: ResolvedMacro = ResolvedMacro.from_config(self.game)
        #: Economy facts handed to the strategist on the next proposal. Set in
        #: ``_refresh_strategy`` so ``strategy_summary`` can include them without
        #: re-reading the state twice.
        self.macro_state: dict[str, Any] = {}
        # Resolved macro plan: the budgets and stage order the curriculum stages read.
        #
        # The stages reference ``self.macro.<field>`` instead of
        # ``self.game.economy_<field>`` so a commander directive (LLM or rule layer) can
        # override them for one episode. With no directive this is byte-identical to the
        # configured defaults -- see ``macro.ResolvedMacro.from_config``.
        #
        # This attribute MUST exist before the first ``step``: every stage dereferences
        # it, so a missing initialisation is not a degraded mode, it is an
        # ``AttributeError`` on the first decision. It is set here (and refreshed in
        # ``reset``) rather than lazily so a direct ``_issue_*`` call on a
        # partially-built env still works.
        self.macro: Any = resolve_macro(self.game, None)

        self._process: _ProcLike | None = None
        self._socket: socket.socket | None = None
        self._reader: _LineReader | None = None
        self._log_handle = None
        self._episode_counter = 0
        self.episode_id = ""
        self.state: dict[str, Any] = {}
        self.masks: dict[str, np.ndarray] = {}
        self.last_tick = 0
        self.last_potential = 0.0
        self.last_distance = 0.0
        # Previous **combat**-centroid gap, kept alongside (never instead of)
        # ``last_distance``. ``engagement`` differences two consecutive readings,
        # so both endpoints have to follow the same centroid definition or the
        # subtraction leaves a constant offset instead of telescoping: a builder
        # wandering on Lake shifts the whole-army centroid by 843 px, worth
        # ``0.3 * 843 / 3677 = +0.0688`` of phantom engagement per episode. The
        # whole-army ``last_distance`` stays untouched for every other consumer.
        # ``None`` means "no attack-capable unit on one side", which the reward
        # reads as no combat signal rather than falling back to the army centroid.
        self.last_combat_distance: float | None = None
        # How many bridge observations were drained-and-dropped because the env
        # could not keep up. A dropped frame is a decision the policy never made,
        # and the drop rate silently changes the control period: the real training
        # configuration (8 envs, frame_ms=1) consumed only 14% of frames, i.e. one
        # decision per ~70 ticks instead of the 10-tick contract, while a
        # 4-worker evaluation consumed ~52% (19 ticks). Exposing the counter keeps
        # that visible instead of letting train and eval disagree in silence.
        self.skipped_frames = 0
        # Whether the opening curriculum is active. Kept as one switch so a run can
        # disable the whole economy ramp without touching three flags.
        self._curriculum_disabled = False
        # The curriculum order sent on the most recent ``step()`` (for logging and
        # for tests); ``None`` when that decision needed no curriculum help.
        self.curriculum_order: dict[str, Any] | None = None
        # Latched build site (grid cell) for the factory curriculum. Chosen once per
        # episode: recomputing it every decision makes the site drift with the
        # builder and the factory restarts forever.
        self.factory_site: tuple[int, int] | None = None
        # Exact world point for the latched factory site. Kept alongside the grid cell
        # because the 8x8 grid quantises to ~364 px on a 2600 px map, which is enough
        # to move a legal land cell into the lake (measured: the dispatched coordinate
        # was (1117.71, 1775.43) -- open water -- while the chosen candidate was land).
        self.factory_exact: tuple[float, float] | None = None
        # Latched turret site, for the same reason: a moving target re-orders the
        # structure every decision and repeating a build order restarts it.
        self.defence_site: tuple[int, int, int, int] | None = None
        # Exact world point for the current turret candidate (see ``_issue_defence``).
        # Kept separately from the grid cell because the 8x8 grid quantises to ~364 px
        # on a 2600 px map, which is enough to move a legal cell into the lake.
        self.defence_exact: tuple[float, float] | None = None
        # Consecutive decisions a turret site has been ordered without a turret
        # appearing; advancing the candidate ladder past ``STALL_DECISIONS``.
        self._defence_stalled: int = 0
        # Turret / factory orders issued this episode. Both stages are bounded because
        # a *rejected* build site produces no hull, so no stall can be observed and the
        # stage would otherwise re-order every decision for the whole game. Measured:
        # 1736 orders and 4/4 timeouts (``.runtime/dispatch_tap.json``).
        self._defence_attempts: int = 0
        self._factory_orders: int = 0
        # Upgrade attempts this episode (see ``reset`` for why attempts, not wins).
        self._upgrades_issued: int = 0
        # Last observed build progress per stage, so a stage can tell "the builder is
        # working, leave it alone" from "the policy cancelled it, re-issue".
        self._extractor_progress: float | None = None
        self._factory_progress: float | None = None
        # Stall counters for the opening curriculum. A build order is only repeated
        # after ``STALL_DECISIONS`` decisions with no forward progress, because
        # repeating one restarts the construction (see :data:`STALL_DECISIONS`).
        self._extractor_stalled = 0
        self._factory_stalled = 0
        self._attack_rotation = 0
        self._aim_rotation = 0
        # Opening-curriculum latches, initialised here as well as in ``reset()``:
        # the stage methods read them, and a caller that drives those methods
        # directly (a test, or a code path that runs before the first reset) must
        # not hit ``AttributeError``.
        self.production_curriculum_done = False
        self.factory_curriculum_done = False
        # Previous economy rate, so the ``income`` reward term can pay for the
        # *growth* of income (a new extractor / an upgrade) instead of paying
        # the absolute rate on every one of thousands of decisions.
        self.last_income_rate = 0.0
        # Previous own-building count: the build reward is paid for a *finished*
        # structure, never for an accepted order (which a policy can repeat
        # every decision to farm reward).
        self.last_own_buildings = 0
        # Previous own *combat* unit count (buildings and builders excluded), so
        # ``bonus_produce`` can pay for a unit actually appearing instead of for
        # a queued order.
        self.last_own_combat = 0
        self.last_own_tier_total = 0
        self.last_own_defences = 0
        self.previous_own_alive = 0
        self.previous_enemy_alive = 0
        # Unit snapshots for threat-weighted ``kills``/``losses`` (see step()).
        self.previous_own_units: list[dict[str, Any]] = []
        self.previous_enemy_units: list[dict[str, Any]] = []
        self.episode_components: dict[str, float] = {}
        self.receipts: list[dict[str, Any]] = []
        self.done = False
        self.last_result: str | None = None
        self.episode_steps = 0
        self.skipped_frames_at_reset = 0
        self.observation_space = _BoxSpace((self.micro.flat_obs_dim,))
        self.action_space = _MultiDiscreteSpace(self.micro.action_nvec)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.seed = int(seed)
        self._episode_counter += 1
        self.episode_id = f"{self.episode_prefix}-{self.seed}-{self._episode_counter}"
        self._shutdown_game()
        self._start_game()
        message = self._await_observation(self.game.start_timeout + self.game.obs_timeout)
        if message is None:
            raise RwppEnvError(f"no observation from real game on port {self.port}")
        self.state = observation_to_state(message)
        # ``state`` stays the ground truth (used for termination / hp / reward);
        # ``obs_state`` is what the policy is allowed to see when fog is enabled.
        self.obs_state = self._observed(self.state)
        self.last_tick = int(self.state["tick"])
        self.masks = build_masks(self.obs_state, self.micro)
        # Reference totals: dead units vanish from the bridge stream, so reward and
        # evaluation fractions are measured against the episode-start army value.
        # Computed from ``self.state`` (ground truth: the bridge streams every
        # unit), NOT from ``self.obs_state`` (fogged policy view), otherwise a
        # fogged episode would normalise enemy hp against the few visible units.
        self.own_max_ref, self.enemy_max_ref = army_hp_reference(self.state, self.micro)
        # ``potential`` rewards the *combat value* difference, not the raw hp
        # difference: units fight at full strength until destroyed, so chip
        # damage across many units must not outscore a kill (see
        # ``reward.combat_value``).
        self.last_potential = combat_value_potential(
            self.state, self.own_max_ref, self.enemy_max_ref
        )
        self.last_distance = army_distance_norm(
            self.state, float(np.hypot(self.micro.map_width, self.micro.map_height))
        )
        self.last_combat_distance = combat_army_distance_norm(
            self.state, float(np.hypot(self.micro.map_width, self.micro.map_height))
        )
        self.last_income_rate = float(
            (self.state.get("global_state") or {}).get("income") or 0.0
        )
        self.last_own_buildings = sum(
            1 for unit in self.state.get("own_entities", []) if unit.get("is_building")
        )
        self.last_own_combat = sum(
            1
            for unit in self.state.get("own_entities", [])
            if not unit.get("is_building") and not unit.get("is_builder")
        )
        # Tier total and defensive-structure count, so ``bonus_upgrade`` and
        # ``bonus_defence`` can be paid as one-off *changes* rather than per step.
        # Both read 0 on a payload that does not report the fields, which keeps the
        # historical reward bit-identical on the mock engine and older bridges.
        #
        # **Buildings only, and tier above 1.** Both restrictions are load-bearing:
        #
        # * every *unit* reports ``tech_level`` 1 (a scout is a T1 scout), so summing
        #   over all entities made "produce a unit" indistinguishable from "upgrade a
        #   building". Measured on the two-slot layout at difficulty 1: ``bonus_upgrade``
        #   was paid on **472 of 498 decisions** while **no building was ever above
        #   T2** -- each new scout added 1 to the total and the policy was paid
        #   ``upgrade_bonus`` (0.15) for it *on top of* ``bonus_produce`` (0.10). The
        #   agent duly spent 15 updates massing scouts with zero economy
        #   (``.runtime/tier_reward_audit.json``);
        # * a building that starts at tier 1 is not an achievement, so only the tiers
        #   *above* the starting one are counted. Otherwise the command centre pays
        #   ``upgrade_bonus`` once at episode start for existing.
        self.last_own_tier_total = _building_tier_total(self.state.get("own_entities"))
        self.last_own_defences = sum(
            1
            for unit in self.state.get("own_entities", [])
            if isinstance(unit, dict)
            and unit.get("is_building")
            and unit.get("is_defence")
            # Exclude the main building and factories: ``is_defence`` means "can
            # shoot", and a command centre reports it (range 280). Counting it would
            # mean ``bonus_defence`` never fires for the first real turret, because
            # the counter already read 1 at episode start.
            and not unit.get("is_factory")
            and not unit.get("is_factory_building")
        )
        self.previous_own_alive = sum(
            1 for unit in self.state.get("own_entities", []) if float(unit.get("hp") or 0.0) > 0.0
        )
        self.previous_enemy_alive = sum(
            1
            for unit in self.state.get("visible_enemy_entities", [])
            if float(unit.get("hp") or 0.0) > 0.0
        )
        self.episode_components: dict[str, float] = {}
        self.receipts = []
        self.done = False
        self.last_result = None
        self.episode_steps = 0
        self._refresh_strategy()
        self.last_strategy_potential = self._strategy_potential()
        # Re-resolve the macro plan now that this episode's directive exists.
        #
        # ``__init__`` resolves against ``None`` (deployment defaults) so the stage
        # methods never see a missing attribute; this refresh applies whatever the
        # strategy layer decided for *this* episode. ``resolve_macro`` is total -- a
        # missing or malformed directive yields the configured defaults -- so this can
        # never leave the env without a plan.
        self.macro = resolve_macro(self.game, self.directive)
        # Opening curriculum: nothing is sent from ``reset()``.
        #
        # The order must travel in the **same dispatch** as the policy's action, not
        # ahead of it. A build placed here is cancelled by the policy's first real
        # order before the builder accumulates any progress (measured on a
        # masked-random Lake episode: ``extractor_ever_present = false``,
        # ``bonus_build_final = 0.0``, 69 cancelling orders in the first 120
        # decisions). ``step()`` collects the curriculum's orders through
        # ``_curriculum_actions()`` and appends them after the policy's own, so the
        # builder keeps building while the rest of the army obeys the policy.
        self.opening_receipt: dict[str, Any] | None = None
        # Production curriculum latch: once we own a finished factory the env
        # queues one combat unit for the policy. See
        # ``RwppGameConfig.opening_production`` for why exploration cannot find
        # this on its own and why it is still paid only by the normal rules.
        self.production_curriculum_done = False
        # Factory curriculum latch: once the opening extractor is finished the env
        # queues one land factory. Without it ``_issue_opening_production`` has no
        # finished factory to queue at, so ``bonus_produce`` can never fire.
        self.factory_curriculum_done = False
        # Reset the latched build site so every episode picks its own (derived from
        # where the builder actually starts, not from the previous game's map side).
        self.factory_site = None
        # Exact world point for the latched factory site; see
        # ``_issue_opening_factory`` for why the grid cell is not enough.
        self.factory_exact = None
        self._extractor_progress = None
        self._factory_progress = None
        self._extractor_stalled = 0
        self._factory_stalled = 0
        # Factory-site rotation. Reset each episode because a site that is illegal on
        # one map need not be illegal on the next.
        self._factory_orders_at_site = 0
        self._factory_dead_sites = set()
        self._factory_candidate_offset = 0
        # Expansion-pool rotation. A pool whose extractor the engine drops never
        # becomes "occupied", so the stage must advance past it or it re-orders the
        # same dead pool forever (measured: 1 occupied pool for 1991 of 2006
        # decisions while 8 were free).
        self._expansion_pool_offset = 0
        self._expansion_ordered_at = None
        self._expansion_occupied = None
        # Distance from the builder to the pool it was last sent to. This is the
        # observable that tells "still walking" apart from "arrived and refused":
        # ``occupied`` alone conflates them, and conflating them is what made the
        # stage switch pools mid-walk (200-tick window vs a 1147-tick walk) and
        # never finish a second extractor.
        self._expansion_distance = None
        self._expansion_stalled = 0
        self._attack_rotation = 0
        self._aim_rotation = 0
        # Upgrade / defence curriculum latches.
        #
        # ``_upgrades_issued`` counts *attempts*, not successes, on purpose: a
        # rejected upgrade would otherwise be retried every single decision forever,
        # and each retry pays ``invalid_penalty``. Counting the attempt bounds the
        # stage's exposure to ``economy_max_upgrades`` rejections per episode, which
        # is cheap, while the reward stays honest because ``bonus_upgrade`` is only
        # paid when ``tech_level`` actually rises.
        self._upgrades_issued = 0
        # Latched turret site, for the same reason as ``factory_site``: a moving
        # target re-orders the structure every decision and repeating a build order
        # restarts construction. The tuple is
        # ``(grid_x, grid_y, turret_count_when_latched, candidate_index)`` because the
        # site must be re-resolved per turret and must be able to rotate away from a
        # cell the engine refuses (a shoreline).
        self.defence_site = None
        # Exact world point for the current turret candidate; see ``_issue_defence``.
        self.defence_exact = None
        self._defence_stalled = 0
        self._defence_attempts = 0
        self._factory_orders = 0
        self.curriculum_order = None
        obs = encode_observation(self.obs_state, self.micro)
        info = {
            "episode_id": self.episode_id,
            "seed": self.seed,
            "masks": self.masks,
            "raw_state": self.state,
            "visible_enemies": len(self.obs_state.get("visible_enemy_entities", [])),
            "tick": self.state["tick"],
            "max_refs": {
                "own": float(self.own_max_ref),
                "enemy": float(self.enemy_max_ref),
            },
            "port": self.port,
        }
        return obs, info

    def step(self, action: np.ndarray):
        if self._socket is None or self._reader is None:
            raise RwppEnvError("reset() must be called before step()")
        self.episode_steps += 1
        previous_potential = float(self.last_potential)
        previous_distance = float(self.last_distance)
        previous_combat_distance = self.last_combat_distance
        previous_income_rate = float(self.last_income_rate)
        previous_own_buildings = int(self.last_own_buildings)
        previous_own_combat = int(self.last_own_combat)
        previous_own_tier_total = int(self.last_own_tier_total)
        previous_own_defences = int(self.last_own_defences)
        previous_own_alive = int(self.previous_own_alive)
        previous_enemy_alive = int(self.previous_enemy_alive)
        receipts_before = len(self.receipts)
        # The opening curriculum's orders ride along in *this* dispatch, after the
        # policy's own orders, so the builder's in-flight construction survives the
        # policy's tactical commands ("last order wins", exactly as for a human).
        # Computed before the send because it must be part of the same message; the
        # reward for this step is still computed from the state the bridge returns,
        # so the order cannot pay into its own step.
        curriculum_actions = self._curriculum_actions()
        # Decode against what the policy actually saw: with fog enabled the
        # target slots index the *visible* enemy list (as masked in obs_state),
        # so using the ground-truth list here would fire at the wrong enemy.
        actions, disconnected = send_step(self, action, curriculum_actions)

        # Always read, even when the send failed.
        #
        # The bridge is a free-running simulator: it advances on ``--frame-ms``
        # and emits an observation every ``DECISION_INTERVAL`` ticks without
        # waiting for a client ``step``. When the engine decides the match it
        # writes ``done`` and closes the socket, so our next ``sendall`` fails
        # with ``ConnectionResetError(10054)`` and the kernel *discards the
        # unread receive queue* -- verdict included.
        #
        # The old code turned that into ``message = None if disconnected else ...``:
        # the read was skipped entirely, the episode was booked as a timeout with
        # ``terminal = 0.0``, and the trainer reported ``win_rate = 0.0`` for all
        # 40 updates of a run whose bridge logs showed 38 victories and 5 defeats.
        #
        # ``_await_observation`` now also drains whatever already arrived and
        # prefers a decisive ``result``/``done`` over a stale frame, so the
        # verdict survives. After a real drop nothing is buffered and the result
        # is still an honest ``timeout`` (``_drain_verdict`` never invents one).
        message = self._await_observation(self.game.obs_timeout)
        if (message is None or not message.get("result")) and disconnected:
            # The step budget may have been eaten by the send failure + catch-up;
            # give the verdict one more bounded chance to surface.
            message = self._drain_verdict() or message
        receipts_this_step = self.receipts[receipts_before:]
        if message is None:
            disconnected = True
            message = {}
        # The bridge reports the native game result on the final observation and
        # on the ``done`` message; it is authoritative when present.
        bridge_result = message.get("result")
        state = absorb_message(self, message)
        if bridge_result is None:
            bridge_result = state.get("result")

        distance = army_distance_norm(
            self.state, float(np.hypot(self.micro.map_width, self.micro.map_height))
        )
        combat_distance = combat_army_distance_norm(
            self.state, float(np.hypot(self.micro.map_width, self.micro.map_height))
        )
        potential = combat_value_potential(self.state, self.own_max_ref, self.enemy_max_ref)
        own_alive = sum(1 for unit in self.state.get("own_entities", []) if float(unit.get("hp") or 0.0) > 0.0)
        enemy_alive = sum(
            1 for unit in self.state.get("visible_enemy_entities", []) if float(unit.get("hp") or 0.0) > 0.0
        )
        previous_strategy = self.last_strategy_potential
        self._refresh_strategy()
        strategy_now = self._strategy_potential()
        strategy_component = (
            float(self.game.strategy_coef) * (strategy_now - previous_strategy)
            if strategy_now is not None and previous_strategy is not None
            else 0.0
        )
        self.last_strategy_potential = strategy_now
        # ---- reward decomposition (docs 7.2/7.4) --------------------------
        shaping = compute_step_shaping(
            micro=self.micro,
            game=self.game,
            state=self.state,
            actions=actions,
            receipts=receipts_this_step,
            message=message,
            bridge_result=bridge_result,
            potential=potential,
            previous_potential=previous_potential,
            distance=distance,
            previous_distance=previous_distance,
            own_alive=own_alive,
            enemy_alive=enemy_alive,
            previous_own_alive=previous_own_alive,
            previous_enemy_alive=previous_enemy_alive,
            previous_own_units=self.previous_own_units,
            previous_enemy_units=self.previous_enemy_units,
            strategy_component=strategy_component,
            disconnected=disconnected,
            done=self.done,
            own_max_ref=self.own_max_ref,
            enemy_max_ref=self.enemy_max_ref,
            episode_components=self.episode_components,
            previous_income_rate=previous_income_rate,
            previous_own_buildings=previous_own_buildings,
            previous_own_combat=previous_own_combat,
            previous_own_tier_total=previous_own_tier_total,
            previous_own_defences=previous_own_defences,
            combat_distance=combat_distance,
            previous_combat_distance=previous_combat_distance,
        )
        components = shaping.components
        reward = shaping.reward
        terminated = shaping.terminated
        truncated = shaping.truncated
        # The opening curriculum is *not* advanced here: its orders ride along in
        # the dispatch that ``_curriculum_actions()`` built at the top of this step
        # (see that method for why a second message cannot work). What matters for
        # reward honesty is that it was computed from the state the bridge returned
        # for this step, so a curriculum order can never pay into its own step --
        # ``bonus_build``/``bonus_produce`` are paid from the observation stream
        # when a structure finishes or a new combat unit appears, at least one
        # decision later.
        if shaping.last_result is not None:
            self.last_result = shaping.last_result
        self.last_potential = potential
        self.last_distance = distance
        self.last_combat_distance = combat_distance
        # A terminal ``done`` message has no ``global_state``; keep the last
        # known rate instead of resetting it to 0, otherwise the next step books
        # a phantom -rate "economy collapse".
        state_global = self.state.get("global_state") or {}
        if "income" in state_global:
            self.last_income_rate = float(state_global.get("income") or 0.0)
        self.last_own_buildings = sum(
            1 for unit in self.state.get("own_entities", []) if unit.get("is_building")
        )
        self.last_own_combat = sum(
            1
            for unit in self.state.get("own_entities", [])
            if not unit.get("is_building") and not unit.get("is_builder")
        )
        self.last_own_tier_total = _building_tier_total(self.state.get("own_entities"))
        self.last_own_defences = sum(
            1
            for unit in self.state.get("own_entities", [])
            if isinstance(unit, dict)
            and unit.get("is_building")
            and unit.get("is_defence")
            # Same exclusion as in ``reset``: the main building reports
            # ``is_defence`` because it has a weapon, and it is not a turret.
            and not unit.get("is_factory")
            and not unit.get("is_factory_building")
        )
        self.previous_own_alive = own_alive
        self.previous_enemy_alive = enemy_alive
        # Snapshot the *units*, not just the counts: ``kills``/``losses`` price a
        # casualty by what it was (``reward.threat_value``), and a count cannot say
        # whether the thing that died was the enemy's artillery or a builder. Only
        # the deltas are used, so the copies are shallow on purpose.
        self.previous_own_units = [dict(unit) for unit in (self.state.get("own_entities") or [])]
        self.previous_enemy_units = [
            dict(unit) for unit in (self.state.get("visible_enemy_entities") or [])
        ]
        if terminated or truncated:
            self._shutdown_game()

        obs = encode_observation(self.obs_state, self.micro)
        info = build_step_info(
            self,
            {
                "actions": actions,
                "bridge_result": bridge_result,
                "disconnected": disconnected,
                "own_alive": own_alive,
                "enemy_alive": enemy_alive,
                "distance": distance,
                "components": components,
                # This decision's receipts only. The rejection histogram in
                # ``build_step_info`` reads it, and it is the only way to tell
                # *why* ``invalid`` was paid (see ``reward_shaping``).
                "receipts": receipts_this_step,
            },
        )
        return obs, reward, terminated, truncated, info

    def _observed(self, state: dict[str, Any]) -> dict[str, Any]:
        """State as the policy sees it (fog of war applied when configured)."""

        radius = float(self.game.vision_radius or 0.0)
        if radius <= 0.0:
            return state
        return apply_vision(dict(state), radius)

    # ------------------------------------------------------------------
    # Strategic layer (LLM or rule based; reward shaping only)
    # ------------------------------------------------------------------
    # The implementation lives in :mod:`ironzero.rwpp.strategy`; these methods
    # stay as delegates so the historical (private) names keep working.
    def _strategy_summary(self) -> dict[str, Any]:
        return strategy_summary(self)

    def _refresh_strategy(self) -> None:
        refresh_strategy(self)

    def _strategy_potential(self) -> float | None:
        return strategy_potential(self)

    def _issue_opening_extractor(self) -> list[dict[str, Any]]:
        """Queue one extractor with our builder (curriculum stage 1).

        Issued through the normal bridge path so the game validates it exactly like
        any policy order. Returns the receipt-ish dict, or ``None`` when the
        curriculum is disabled / there is no builder / no free resource pool -- all
        of which are ordinary situations on maps that are not economically
        playable, so this must never raise.

        Deliberately *not* rewarded directly: ``bonus_build`` and ``income`` are
        computed from the observation stream and only pay when the extractor
        actually finishes and starts generating. The point is to remove the
        exploration barrier (a build needs ~140 uninterrupted ticks), not to hand
        out free reward.

        Called from ``reset()`` **and** from :meth:`_run_opening_curriculum` on
        every subsequent decision until the extractor exists. Re-issuing is the
        load-bearing part: a one-shot order placed at reset is cancelled by the
        policy's very first real order (measured on a masked-random Lake episode:
        ``extractor_ever_present = false``, ``bonus_build_final = 0.0``, and 69 of
        the first 120 decisions carried a cancelling order). Measured cost of
        re-issuing is bounded -- the factory still completes, at tick 2611 versus
        1891 for a single order.
        """

        if not getattr(self.game, "opening_extractor", False):
            return []
        state = self.state
        own = state.get("own_entities", [])
        # Re-issue until the pool extractor is **finished**, not merely started.
        #
        # Both halves of this condition were measured and both are load-bearing:
        #
        # * stopping when the entity merely *exists* (the first draft) handed a
        #   half-built extractor back to the policy, which cancelled it on the next
        #   real order and the stage never re-issued. On a masked-random Lake
        #   episode the extractor appeared at tick 1731 and did not finish until
        #   tick 17241 -- 1551 decisions of a 140-tick build;
        # * never stopping (no latch at all) re-issued ``build_extractor`` every
        #   decision, which cancels the *factory* ordered by stage 2, and the
        #   episode ended at tick 20056 with ``extractorT1`` as the only structure.
        #
        # So the stage owns the build until it is complete, then gets out of the
        # way. Measured cost of re-issuing: a factory ordered every decision still
        # completes, at tick 2611 versus 1891 for one uninterrupted order (~1.4x).
        # "Done" means *operational*, not ``build_progress >= 1.0``.
        #
        # The two disagree on purpose, and the difference decides this stage. A
        # T1 extractor that has finished can be upgraded to T2/T3; during the
        # upgrade the engine reports ``build_progress`` below 1.0 again while
        # ``is_operational`` stays true (the building still produces). Gating on raw
        # progress therefore made the stage try to "finish" an extractor that the
        # policy kept upgrading: measured on a masked-random Lake episode, the
        # extractor reached ``extractorT3`` with ``income`` 1.0 and the stage still
        # refused to hand over for the rest of the 2006-decision episode, so the
        # factory was never ordered.
        #
        finished_extractor = any(
            "xtractor" in str(unit.get("unit_type") or "") and _is_operational(unit)
            for unit in own
        )
        if finished_extractor:
            # The opening extractor is done; this stage is satisfied. Further
            # extractors are built by :meth:`_issue_late_expansion`, which runs
            # **after** the factory and production stages so it can never take the
            # builder away from them.
            return []
        # Already mid-build: re-issue against the *same* pool so the order keeps the
        # builder working on the structure it started rather than opening a second
        # one elsewhere.
        # The pool is identified by proximity, the same 40 px rule the bridge uses
        # to report ``occupied`` (``IronZeroBridge.slotExtractorLevel``), because a
        # pool with a half-built extractor on it already reads as occupied and is
        # therefore *excluded* from ``free``.
        building = next(
            (
                unit for unit in own
                if "xtractor" in str(unit.get("unit_type") or "")
            ),
            None,
        )
        slots = state.get("resource_slots") or []
        free = [
            index
            for index, slot in enumerate(slots)
            if isinstance(slot, dict) and not slot.get("occupied")
        ]
        slot_id: int | None = None
        if building is not None and slots:
            bx = float(building.get("x") or 0.0)
            by = float(building.get("y") or 0.0)
            best_d = float("inf")
            for index, slot in enumerate(slots):
                if not isinstance(slot, dict):
                    continue
                d = (float(slot.get("x") or 0.0) - bx) ** 2 + (
                    float(slot.get("y") or 0.0) - by
                ) ** 2
                if d < best_d:
                    best_d = d
                    slot_id = index
        if slot_id is None:
            if not free:
                return []
            # Nearest-to-own-base first is how the bridge already orders the list,
            # so the first free pool is the intended opening build.
            slot_id = int(free[0])
        if not any(unit.get("is_builder") for unit in own):
            return []
        # Re-issue only when the build has actually **stalled**.
        #
        # This replaces an earlier "re-issue on every decision" rule, and the
        # change is measured, not stylistic. A/B against the real engine on
        # ``[p2]Lake (2p)``, same seed, a policy that issues nothing
        # (``.runtime/ab_reissue.py``):
        #
        #     mode    orders sent   extractor build_progress at decision 140
        #     once              1   1.000  -> operational at decision 140
        #     every           140   0.000  -> NEVER completes, frozen for 1200 decisions
        #     stall(20)         3   1.000  -> operational at decision 140
        #
        # Repeating ``build_extractor`` for the *same* pool is therefore **not** a
        # no-op: the engine restarts the construction each time, so progress is
        # pinned at exactly 0.0 while every receipt still reports
        # ``accepted: true``. The deadlock is self-sustaining -- this stage refuses
        # to hand over until the extractor is ``is_operational``, and re-issuing is
        # what prevents it from ever getting there. Measured end state of that
        # livelock on a pure-noop episode: ``peak_own = 3`` (command center +
        # builder + a permanently 0% extractor) while the AI reached 15 units, and
        # the factory stage was never called even once in 900 decisions.
        #
        # The earlier note in this docstring claimed the opposite ("re-issuing is
        # what makes last order wins hold"). That was measured under a
        # *masked-random* policy whose own orders kept hitting the builder, where
        # an every-decision order at least re-tasked it. Under any policy that
        # leaves the builder alone -- which includes the curriculum's own intent --
        # it is fatal.
        #
        # The gate below re-orders only after ``STALL_DECISIONS`` consecutive
        # decisions with no forward progress, so a policy that *does* pull the
        # builder away is still recovered from, while an undisturbed build runs at
        # full speed.
        progress_now = None
        if building is not None:
            try:
                progress_now = float(building.get("build_progress") or 0.0)
            except (TypeError, ValueError):
                progress_now = None
        if progress_now is not None:
            previous = getattr(self, "_extractor_progress", None)
            if previous is None or progress_now > float(previous) + 1e-9:
                # Making progress: record it and stay out of the way.
                self._extractor_progress = progress_now
                self._extractor_stalled = 0
                return []
            self._extractor_stalled = int(getattr(self, "_extractor_stalled", 0)) + 1
            if self._extractor_stalled < STALL_DECISIONS:
                return []
            self._extractor_stalled = 0
        action = np.array(
            [ACTION_KINDS.index("build_extractor"), int(self.micro.groups),
             int(self.micro.position_grid) // 2, int(self.micro.position_grid) // 2,
             int(slot_id)],
            dtype=np.int64,
        )
        return self._send_curriculum_order(action, {"kind": "build_extractor",
                                                   "slot_id": int(slot_id)})

    def _send_direct_order(
        self, order: dict[str, Any], payload: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Dispatch an already-decoded order, bypassing the action-vector round trip.

        Some curriculum decisions name a concrete entity ("queue a tank at factory
        27"). Encoding that as a head index and decoding it again is not just
        redundant: the index space is rebuilt from the live observation and changes
        size whenever a factory is busy, so the same index names a different pair one
        decision later. Measured consequence: a land factory was ordered to build a
        unit absent from its menu, the queue item never started (progress -1.0), and
        the stage read ``queue_full`` for 771 of 900 decisions with the army at zero.

        Returns a single-element list so callers keep the same "orders to append"
        contract as :meth:`_send_curriculum_order`.
        """

        if not isinstance(order, dict) or not order.get("kind"):
            return []
        self.curriculum_order = {**payload, "accepted": True}
        return [order]

    def _send_curriculum_order(
        self, action: np.ndarray, payload: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Decode one curriculum action into orders for the current dispatch.

        Returns a single-element list, or an empty list when the action decodes to
        nothing (e.g. a masked build kind). Never sends anything itself: the orders
        are appended to the step the caller is already sending, so a curriculum
        order costs neither an extra observation nor a separate dispatch.
        """

        try:
            orders = self._decode_curriculum_action(action)
        except (ValueError, IndexError):
            return []
        if not orders:
            return []
        self.curriculum_order = {**payload, "accepted": True}
        return orders

    def _decode_curriculum_action(self, action: np.ndarray) -> list[dict[str, Any]]:
        """Decode a curriculum action, whatever action layout is configured.

        Every curriculum stage builds a **five-head** vector (``kind, group, x, y,
        target``) because that is the vocabulary it reasons in: it names one
        concrete order ("build an extractor on pool 2") and never thinks in slots.

        ``decode_action`` validates the vector length against
        ``len(config.action_nvec)``, which under the two-slot layout is **10**. So
        handing a curriculum vector straight to ``decode_action`` raises
        ``ValueError``, the ``except`` above swallows it, and the stage looks like
        it simply chose to do nothing. Measured on Lake with
        ``action_slots=2, curriculum_mode="orders"``: ``curriculum_issues_orders``
        was ``True`` and ``_curriculum_actions()`` returned ``[]`` on **400 of 400**
        decisions, ``peak_extractors = 0`` and income never left the base 18.0 for a
        whole episode (``.runtime/slot_diag_orders_2_noop.json``).

        That failure mode is worth naming because it is silent by construction: the
        curriculum is *supposed* to return ``[]`` most of the time (every stage is
        gated), so "it never fires" and "it always raises" are indistinguishable
        from the orders alone. Decoding against a single-slot view of the config
        fixes it without teaching the stages about slots: ``_decode_heads`` reads
        only ``vector[0..4]``, exactly where the curriculum put its values.
        """

        config = self.micro
        if getattr(config, "is_two_slot", False):
            from dataclasses import replace as _replace

            config = _replace(config, action_slots=1)
        # ``extractor_slot_is_global=True``: every stage resolves a **concrete**
        # pool before it builds its vector ("the pool nearest our base", "the pool
        # a half-built extractor is sitting on"). Head 4 therefore already names a
        # global slot id, and passing it back through the policy's free-pool
        # ordinal space would re-index it: measured with pool 0 occupied, the
        # opening stage asking to *finish* pool 0 was decoded as pool 1 (so it
        # abandoned its own half-built extractor and started a second one), and
        # the expansion stage asking for pool 2 was decoded as pool 3.
        return decode_action(
            action, self.obs_state, config, extractor_slot_is_global=True
        )

    def _issue_upgrades(self) -> list[dict[str, Any]]:
        """Spend credits on a tier upgrade before spending them on another unit.

        The ordering is the whole content of this stage, so it is stated up front:
        a T1 -> T2 extractor is a *permanent* increase in the rate that pays for
        every future unit, while one more tank is a one-off. Rusted Warfare's economy
        compounds, so upgrading first is the correct economic priority.

        Why the environment has to do this at all
        ----------------------------------------
        Nothing in the system ever upgraded anything, and both halves were measured:

        * the policy never sampled ``upgrade_building`` usefully. It could -- the
          mask opened the kind whenever the bridge reported ``can_upgrade`` -- but the
          reward had no term for it (``income`` only pays for the rate growth after
          the fact, which arrives hundreds of ticks later, long after the credit was
          spent on the action that produced it);
        * the bank was never available even if it had. ``economy_credit_reserve`` is
          what the production stage refuses to spend below, and an upgrade costs
          1200 against a 4000 starting bank -- so production drained every credit
          above the reserve down to the reserve and the upgrade stayed unaffordable
          for the entire episode.

        So the stage runs **before** production in ``_curriculum_actions``'s priority
        list: it claims the credits first, and production only spends the remainder.
        The reciprocal gate below is what stops both from fighting over the same
        bank -- this stage refuses to act unless a real surplus exists, so it can
        never starve the army to death.

        Every order travels the normal bridge path and ``bonus_upgrade`` is paid by
        the observation stream when ``tech_level`` actually increases, so repeating
        the order cannot farm reward.
        """

        if not getattr(self.game, "economy_upgrade", False):
            return []
        if not self.macro.allows("upgrade"):
            # The commander cancelled the tier ladder for this decision (e.g. a
            # broken army: a 1200-credit rate increase pays back too late).
            return []
        if int(self._upgrades_issued) >= int(self.macro.max_upgrades):
            return []
        state = self.state
        upgradable, reported = upgradable_buildings(state)
        if not upgradable:
            # ``reported`` false means an older bridge that cannot answer; issuing an
            # upgrade there is a coin flip, and we have no ``tech_level`` stream to
            # reward it with either, so stay out.
            return []
        # Only a *finished* building upgrades: the bridge already gates
        # ``can_upgrade`` on ``isNonOperational()``, but re-checking here keeps the
        # stage honest if an older bridge reports the raw action id.
        upgradable = [
            unit for unit in upgradable if _is_operational(unit)
        ]
        if not upgradable:
            return []
        own = state.get("own_entities") or []
        # NOTE: there is deliberately **no** "builder must be idle" gate here.
        #
        # There used to be one, on the theory that the engine routes a build-queue
        # item through the builder and an order would interrupt an in-flight
        # construction. That is true of ``build_structure``/``build_extractor``,
        # which address the *builder*, but not of an upgrade: the bridge's
        # ``upgradeBuilding`` creates a command for the building itself
        # (``engine.commandController.createCommandForTeam`` +
        # ``command.addUnit(orderable)``, IronZeroBridge.java:3557) and dispatches
        # the building's own ``getBuildActionId()`` special action. No builder is
        # involved, so an in-flight construction is not disturbed.
        #
        # The gate was not cosmetic: combined with the ``combat_alive >= 1``
        # precondition below it made the stage unreachable in practice. Measured on
        # Lake at difficulty 1 (scripts/probe_upgrade_gates.py), the builder has a
        # structure under construction on essentially every decision from tick 0 to
        # the end, so the two conditions never held together and
        # ``_issue_upgrades`` emitted **zero** orders across 1200 decisions with
        # five upgradable buildings, max_upgrades=4 and a 4004-credit opening bank.
        # Every building finished the episode at tech_level 1 while the enemy
        # fielded extractorT2, heavyInterceptor (shield 450) and mechFactory units.
        # Leave enough for the army.
        #
        # Two separate floors, because they protect different things:
        #
        # * ``economy_credit_reserve`` is what the *production* stage refuses to
        #   spend below, so requiring it here means an upgrade can never leave the
        #   builder unable to finish the next structure;
        # * ``extractor_upgrade_cost`` is the upgrade itself.
        #
        # Requiring both makes the gate 2100 on the default config. That was measured
        # to be too strict in practice (``.runtime/curriculum_e2e.json``: a whole
        # episode produced exactly **one** upgrade, because production drains the bank
        # to ~400-800 and it rarely climbs back over 2100). The stage now runs first,
        # so it sees the bank *before* production spends it, and a lighter floor lets
        # it act more than once per game while still leaving the reserve intact.
        #
        # Do not upgrade at the cost of having no army at all.
        #
        # This must be a *precondition*, not a target. An earlier version required
        # ``economy_army_target // 2`` (= 5) living combat units, and the measured
        # consequence was that the stage never fired: on a fast episode the army peaks
        # at exactly 5 and the game ends before a sixth exists
        # (``.runtime/multiseed.json``: three seeds, ``max_combat = 5`` in all three,
        # ``upgraded = 0``). Gating a permanent rate increase behind an army size the
        # episode may never reach is the same mistake as requiring the reserve *plus*
        # the cost -- measured earlier in ``.runtime/upgrade_blockers.json`` where that
        # combination blocked the stage on 133 of 448 decisions.
        #
        # The honest precondition is "an army *source* exists and is working": an
        # operational factory plus one living combat unit. That is enough to know the
        # economy is not being traded away for a tier, without depending on a unit
        # count that the episode length controls.
        combat_alive = sum(
            1
            for unit in own
            if unit.get("can_attack")
            and not unit.get("is_builder")
            and not unit.get("is_building")
            and float(unit.get("hp") or 0.0) > 0.0
        )
        has_working_factory = any(
            (unit.get("is_factory") or unit.get("is_factory_building"))
            and _is_operational(unit)
            for unit in own
        )
        if not has_working_factory or combat_alive < 1:
            return []
        # Credits: the upgrade plus a small margin, never below the production
        # reserve. ``need`` is derived from the *price list* rather than hard-coded so
        # it tracks ``extractor_upgrade_cost`` if that changes.
        funds = float((state.get("global_state") or {}).get("funds") or 0.0)
        # The engine's upgrade price is 1200, so the old gate
        # ``extractor_upgrade_cost * 1.1`` = 1320 was never reachable: measured on
        # Lake at difficulty 1 (scripts/probe_upgrade_gates.py) the bank oscillates
        # 700-1054 and ``_issue_upgrades`` emitted **zero** orders across 1200
        # decisions even with five upgradable buildings and max_upgrades=4.
        #
        # The reserve*0.5 floor (350) is the real protection: the production stage
        # has already claimed its reserve before this stage sees the bank, so the
        # only question is whether the upgrade itself can be paid for. Asking for
        # the raw price (not price*1.1) lets it fire the moment 1200 is available,
        # which does happen during the opening when the starting bank is 4000.
        need = max(
            float(self.macro.credit_reserve) * 0.5,
            float(self.micro.extractor_upgrade_cost),
        )
        if funds < need:
            return []
        # Prefer the *cheapest tier first* -- i.e. a T1 building over an already-T2
        # one -- because the marginal credit is worth more at the bottom of the
        # ladder. ``tech_level`` is missing on older payloads, where every building
        # sorts equal and the first one wins, which matches the previous behaviour.
        building = min(
            upgradable,
            key=lambda unit: float(unit.get("tech_level") or 1.0),
        )
        action = np.array(
            [ACTION_KINDS.index("upgrade_building"), int(self.micro.groups),
             int(self.micro.position_grid) // 2, int(self.micro.position_grid) // 2,
             int(self._upgrade_target_slot(upgradable, building))],
            dtype=np.int64,
        )
        self._upgrades_issued += 1
        return self._send_curriculum_order(
            action,
            {"kind": "upgrade_building",
             "building_entity_id": int(building.get("entity_id", -1))},
        )

    @staticmethod
    def _upgrade_target_slot(upgradable: list[dict[str, Any]], chosen: dict[str, Any]) -> int:
        """Index of ``chosen`` in the list the decoder will index.

        The decoder selects the upgrade target with ``upgradable[target % len]``
        from ``menus.upgradable_buildings``, so the curriculum must encode an index
        into that *same* list or it would ask for a different building than the one
        it chose. Returns 0 when the list is degenerate, which reproduces the old
        fixed-0 behaviour.
        """

        try:
            return upgradable.index(chosen)
        except ValueError:
            return 0

    def _issue_defence(self) -> list[dict[str, Any]]:
        """Place a defensive structure between our base and the enemy.

        A turret is the only building that protects the income paying for everything
        else, and the agent never built one. That was not a learning failure: until
        ``build_structure`` could name a *structure type*, the action space had no
        way to express "turret" at all -- every such order meant "another land
        factory". Measured in ``.runtime/buildnames.json``: this very map's builder
        accepts ``structure_type="turret"``, so the capability was never in doubt.

        Placement is on the *own-base side of the lake*, at the point where the two
        bases face each other. That is where the AI's pushes arrive on this map, and
        a turret anywhere else is a pure cost.

        Deliberately last in credit priority (after upgrades and after production):
        a turret that never fires is a sunk cost, while income and army compound.
        """

        if not getattr(self.game, "economy_defence", False):
            return []
        if not self.macro.allows("defence"):
            return []
        # Bounded attempts per episode, for the same reason the upgrade stage is
        # bounded: a build order that never produces an entity would otherwise be
        # re-issued for the rest of the game, and each re-issue pays
        # ``invalid_penalty`` on a rejection while monopolising the one builder.
        #
        # Measured (``.runtime/trained_behaviour.json``): unbounded, this stage issued
        # ``build_structure`` **941-1163 times per episode**, the army collapsed to a
        # single ``scout``, and four straight wins became four straight timeouts. The
        # cap is generous enough for the configured turret count plus a few failed
        # sites, and small enough that a permanent failure cannot eat the episode.
        max_attempts = int(self.macro.max_turrets) * 8 + 8
        if int(self._defence_attempts) >= max_attempts:
            return []
        state = self.state
        own = state.get("own_entities") or []
        # Count what we already have, by the entity stream rather than by our own
        # bookkeeping, so a turret the policy built also counts towards the limit.
        #
        # The command centre and any factory are excluded, and that exclusion is
        # load-bearing rather than tidy. ``is_defence`` is a *capability* flag ("this
        # building can shoot"), not a category, and the engine reports it for the main
        # building too: measured on Lake (``.runtime/q7d_sea_produce.json``) a
        # ``commandCenter`` carries ``is_defence: true, can_attack: true,
        # attack_range: 280``. Counting it left this stage believing one defensive
        # structure already existed from tick 1, so with ``economy_max_turrets = 2`` it
        # built exactly **one** turret per episode and then stopped for good --
        # measured in ``.runtime/trained_behaviour.json``: ``max_defences`` was 1 in
        # 6 of 6 episodes while the stage itself fired on dozens of decisions.
        #
        # What this stage owns is *static defensive structures it chose to place*, so
        # it counts buildings that can shoot and are neither the main building nor a
        # production facility.
        defences = [
            unit
            for unit in own
            if unit.get("is_building")
            and unit.get("is_defence")
            and not unit.get("is_factory")
            and not unit.get("is_factory_building")
        ]
        if len(defences) >= int(self.macro.max_turrets):
            return []
        # A turret already being built counts as "at the limit for now".
        #
        # Without this the stage re-issues the order every decision for the whole
        # ~1570-tick construction, and every re-issue **restarts** it
        # (``.runtime/ab_reissue.py``: ``build_progress`` pinned at 0.0 forever while
        # each receipt says ``accepted: true``).
        #
        # Measured cost of missing this (``.runtime/trained_behaviour.json``):
        # ``build_structure`` was issued **1821-1843 times in a single episode**,
        # which monopolised the builder, starved the factory and the army down to one
        # ``scout``, and turned four straight wins into four straight timeouts. A
        # structure under construction is not yet counted by the entity stream (it
        # reports ``is_defence`` but the count above uses live buildings), so it must
        # be excluded explicitly.
        if any(
            unit.get("is_building")
            and unit.get("is_defence")
            and not unit.get("is_factory")
            and not unit.get("is_factory_building")
            and not _is_operational(unit)
            for unit in own
        ):
            return []
        # The chain must stand first: a turret is a luxury until there is a factory
        # and an income to defend.
        if not any(unit.get("is_factory") for unit in own if _is_operational(unit)):
            return []
        if not any("xtractor" in str(unit.get("unit_type") or "") for unit in own):
            return []
        # A factory already under construction means the builder is busy, so a turret
        # must not compete for it in this decision.
        #
        # Without this the two stages fight: the defence stage is placed before the
        # factory stage so it can never be starved (a blocking build order ends the
        # priority loop), which means it must itself yield while the factory is being
        # built. Measured both ways in ``.runtime/trained_behaviour.json``: with defence
        # ahead and no yield, factories were built but turrets were 0/6; with defence
        # behind, turrets were 5/6 but factories were 0/6. Yielding on an in-flight
        # *factory* is what lets both happen in one episode.
        if any(
            unit.get("is_building")
            and not _is_operational(unit)
            and (unit.get("is_factory") or unit.get("is_factory_building"))
            for unit in own
        ):
            return []
        # Do not cancel an in-flight build.
        if any(unit.get("is_building") and not _is_operational(unit) for unit in own):
            return []
        # A defensive structure needs money the army does not.
        #
        # The gate used to be ``economy_credit_reserve + turret_cost`` = 1400, and the
        # measurement is unambiguous about what that did: it blocked the stage on
        # **1520 of 1968** decisions (``.runtime/defence_blockers.json``) and the stage
        # reached its only firing opportunity once per episode, which is why
        # ``max_defences`` was 1 in every run.
        #
        # The reason is structural, not a tuning accident: the *production* stage is
        # designed to drain the bank down to ``economy_credit_reserve`` (900). Requiring
        # 900 + 500 therefore asks for a surplus the production stage exists to remove,
        # so the two gates can never both be satisfied for long.
        #
        # A turret is cheaper than a tank and is what protects the income paying for
        # everything else, so the honest gate is "do not spend the production reserve",
        # i.e. keep the reserve intact but let the turret come out of anything above
        # *one* turret's price. That is strictly weaker than the old gate and still
        # cannot starve the builder.
        funds = float((state.get("global_state") or {}).get("funds") or 0.0)
        if funds < float(self.micro.turret_cost) * 1.5:
            return []
        builder = next(
            (unit for unit in own
             if unit.get("is_builder") and float(unit.get("hp") or 0.0) > 0.0),
            None,
        )
        if builder is None:
            return []
        menu = build_menu(state)
        if not menu:
            return []
        type_index = next(
            (index for index, name in enumerate(menu) if is_defence_type(name)),
            None,
        )
        if type_index is None:
            # The builder cannot make a defensive structure on this map. Nothing to
            # order -- and issuing a factory instead would be a lie about intent.
            return []
        # Site: on the line between our base and the enemy, closer to home than to
        # the front, so the turret covers the approach instead of dying alone.
        anchor = next(
            (unit for unit in own if unit.get("is_factory_building")), None
        ) or builder
        ax = float(anchor.get("x") or 0.0)
        ay = float(anchor.get("y") or 0.0)
        foes = state.get("visible_enemy_entities") or []
        if foes:
            ex = sum(float(u.get("x") or 0.0) for u in foes) / len(foes)
            ey = sum(float(u.get("y") or 0.0) for u in foes) / len(foes)
        else:
            # Nothing visible: the enemy is on the opposite side of the map on a
            # two-player skirmish map, which is what the previous fallbacks assumed.
            ex = float(self.micro.map_width) - ax
            ey = float(self.micro.map_height) - ay
        # Candidate sites, nearest-the-base first.
        #
        # A single fixed fraction is not enough, and the reason is measured rather
        # than hypothetical: "25% of the way to the enemy" landed a turret at
        # ``(1117, 1775)`` on Lake, and that cell is **water** -- the lake's south edge
        # sits at y ~1780 (``.runtime/map_water.json``: water bbox y 620..1780 px).
        # With the bridge's new placement gate the order is now honestly rejected
        # instead of silently doing nothing, so the stage would simply never finish a
        # turret: measured ``turret_count_ever = 0`` with ``WOULD_FIRE = 1``
        # (``.runtime/defence_blockers.json``).
        #
        # The curriculum does not have a terrain grid, so it must not pretend to know
        # where the water is. What it *can* do is stay close to its own base -- which
        # is land by construction, since we start there -- and rotate through a small
        # ladder of nearby cells when one is refused. That is map-agnostic: on a map
        # with no water the first candidate simply works, and on a map like Lake the
        # rotation walks back toward the base until it leaves the shoreline.
        # The ladder is ordered by *measured* legality, nearest-the-front first.
        #
        # These fractions were verified against the engine itself
        # (``.runtime/turret_sites.json``), one turret per cell, no model in the loop:
        #
        # <pre>
        #   fraction  world (x, y)      engine result
        #   0.02      ( 994, 1974)      built
        #   0.05      (1002, 1920)      built
        #   0.10      (1014, 1830)      built
        #   0.16      (1028, 1722)      BUILD_UNAVAILABLE   <- lake shoreline
        #   0.22      (1043, 1614)      BUILD_UNAVAILABLE   <- open water
        #   0.25      (1050, 1560)      BUILD_UNAVAILABLE   <- open water
        # </pre>
        #
        # So on Lake the usable band toward the lake is roughly 0.10 and shallower. The
        # earlier ordering put 0.10 first but followed it with *deeper* fractions, which
        # meant the first refusal made the stage walk further into the lake before it
        # came back -- measured ``turret_count_ever = 0`` with the stage firing twice
        # before the episode ended (``.runtime/defence_blockers.json``).
        #
        # Ordering inward (0.10 -> 0.05 -> 0.02) makes every subsequent candidate
        # *more* likely to be legal rather than less, which is the property that
        # matters: a rotation should converge, not wander.
        fractions = (0.10, 0.05, 0.02, 0.14, 0.0)
        grid = int(self.micro.position_grid)
        # Candidates are kept in **exact world coordinates**, not as grid indices.
        #
        # The grid is the *policy's* addressing scheme (its position head has 8
        # cells). The curriculum knows the point it wants, and quantising it loses the
        # very precision that makes the cell legal: measured
        # (``.runtime/turret_sites.json`` vs ``.runtime/defence_blockers.json``), the
        # legal cell (1014, 1830) snaps to grid (3, 5) = (1118, 1775), and (1118,
        # 1775) is water, so the engine refused every order the stage sent while the
        # stage believed it was aiming at a legal spot.
        occupied_now = [
            (float(u.get("x") or 0.0), float(u.get("y") or 0.0))
            for u in own
            if u.get("is_building")
        ]
        candidates: list[tuple[int, int, float, float]] = []
        for fraction in fractions:
            wx = ax + (ex - ax) * fraction
            wy = ay + (ey - ay) * fraction
            # Two buildings cannot share a cell, so skip anything standing on top of an
            # existing structure -- that rejection would have nothing to do with
            # terrain and would waste a whole rotation step.
            if any(
                (wx - ox) ** 2 + (wy - oy) ** 2 < 200.0 * 200.0
                for ox, oy in occupied_now
            ):
                continue
            gx_i = int(round(wx / max(1.0, float(self.micro.map_width)) * (grid - 1)))
            gy_i = int(round(wy / max(1.0, float(self.micro.map_height)) * (grid - 1)))
            candidates.append(
                (
                    max(0, min(grid - 1, gx_i)),
                    max(0, min(grid - 1, gy_i)),
                    wx,
                    wy,
                )
            )
        if not candidates:
            return []
        # The site is latched **per turret**, not for the whole episode.
        #
        # A single episode-wide latch is right for the *factory* stage (re-issuing
        # must hit the same half-built hull) but wrong here: once the first turret
        # occupies its cell, a second turret aimed at the same coordinate can never be
        # placed, so ``economy_max_turrets = 2`` silently degraded to 1.
        #
        # ``defence_site`` is ``(grid_x, grid_y, turret_count_when_latched,
        # candidate_index)`` and ``defence_exact`` holds the exact world point for the
        # current candidate. The candidate index advances only when a turret fails to
        # appear after ``STALL_DECISIONS`` decisions -- the same stall rule the build
        # stages use -- because re-issuing at the same illegal cell forever would lock
        # the stage out for the rest of the episode.
        latched_count = None if self.defence_site is None else self.defence_site[2]
        if self.defence_site is None or latched_count != len(defences):
            self.defence_site = (
                int(candidates[0][0]),
                int(candidates[0][1]),
                len(defences),
                0,
            )
            self.defence_exact = (candidates[0][2], candidates[0][3])
            self._defence_stalled = 0
        gx, gy, _count, candidate_index = self.defence_site
        if self._defence_stalled >= STALL_DECISIONS:
            # This site has been retried long enough without a turret appearing.
            # Advance through the ladder (wrapping) rather than giving up, so a
            # shoreline cell costs a few hundred ticks instead of the whole episode.
            candidate_index = (candidate_index + 1) % len(candidates)
            gx, gy, wx, wy = candidates[candidate_index]
            self.defence_site = (gx, gy, len(defences), candidate_index)
            self.defence_exact = (wx, wy)
            self._defence_stalled = 0
        else:
            self._defence_stalled = int(self._defence_stalled) + 1
        self._defence_attempts = int(self._defence_attempts) + 1
        # Emit the order with the *exact* world coordinate, not through the 8x8 grid.
        #
        # ``grid_to_world`` quantises to ``position_grid`` steps, and on a 2600 px map
        # that is ~364 px per step. Measured (``.runtime/defence_blockers.json``,
        # ``.runtime/turret_sites.json``): the legal turret cell at (1014, 1830) snaps
        # to grid cell (3, 5) = (1118, 1775), and (1118, 1775) is **water**. So the
        # stage kept ordering a legal position and the engine kept refusing the
        # quantised one -- ``turret_count_ever = 0`` while the stage fired on dozens of
        # decisions.
        #
        # The grid exists for the *policy*, whose position head has 8 cells; the
        # curriculum knows the exact point it wants and has no reason to lose it.
        # ``_send_curriculum_order`` decodes an action vector, so the exact coordinate
        # is emitted directly here instead.
        wx, wy = self.micro.grid_to_world(int(gx), int(gy))
        if self.defence_exact is not None:
            wx, wy = self.defence_exact
        self.curriculum_order = {
            "kind": "build_structure",
            "structure_type": menu[type_index],
            "grid": [int(gx), int(gy)],
            "world": [round(float(wx), 2), round(float(wy), 2)],
            "accepted": True,
        }
        return [
            {
                "kind": "build_structure",
                "unit_group": int(self.micro.groups),
                "structure_type": menu[type_index],                "x": round(float(wx), 2),
                "y": round(float(wy), 2),
            }
        ]

    def _issue_attack_drive(self) -> list[dict[str, Any]]:
        """Send the army at the enemy base once it is big enough to matter.

        Why the environment has to do this at all
        ----------------------------------------
        The measured noop baseline is unambiguous. With a complete economy in place
        (5 extractors, ``income`` 50-58) the army reaches 10 combat units and then
        **stands still for the rest of the episode**: measured
        (``.runtime/engagement_trace.json``) the nearest own combat unit to the enemy
        base is 1972 px at tick 9001 and *exactly the same* 1972 px at tick 15001,
        i.e. nothing moved for 60 seconds of game time. Five of six episodes ended
        in a ``timeout`` with both armies alive.

        A timeout is not a near-miss here -- it is a **loss of the objective**.
        ``TriggerEngine``'s stock objective on a skirmish map is
        ``noConstructionOrTech`` plus the engine's own defeat bookkeeping, so a
        battle that is never fought never resolves. Nothing in PPO will discover
        "walk 2600 px and attack" from a reward that also pays for economy, because
        the first several hundred ticks of marching pay nothing.

        What this stage is *not*
        ------------------------
        It is not a scripted army and it does not decide tactics: it issues exactly
        one ``attack_move`` toward the enemy base, at the group the policy is free
        to override on the next decision. Every order still travels the normal
        bridge path, the engine still resolves the fight, and ``kills`` /
        ``potential`` / ``terminal`` are paid by the normal rules. The policy keeps
        full authority -- if it wants to hold, it issues ``hold`` and this stage
        will re-issue one decision later, the same "last order wins" arbitration
        the build stages use.

        Gates:

        * **An army must exist.** ``economy_army_target`` combat units, otherwise
          marching is just feeding units in one at a time. This is deliberately the
          same threshold the production stage sustains.
        * **A target must exist.** The enemy base is located from the observation:
          its command center when visible, else the enemy centroid (the same
          definition ``reward_shaping.combat_army_distance_norm`` uses).
        * **The order repeats while the army is not yet engaged**, and stops once
          the armies are close, so during the fight the policy owns the micro
          decisions instead of being overridden every decision.
        """

        if not getattr(self.game, "opening_attack", False):
            return []
        if not self.macro.allows("attack"):
            # The commander cancelled the march (e.g. the army is broken and the
            # plan is to rebuild instead of feeding units one at a time).
            return []
        state = self.state
        own = state.get("own_entities") or []
        ours = [
            unit
            for unit in own
            if unit.get("can_attack")
            and not unit.get("is_builder")
            and not unit.get("is_building")
            and float(unit.get("hp") or 0.0) > 0.0
        ]
        if len(ours) < int(self.macro.attack_at_army):
            # A stalled army must still march.
            #
            # ``attack_at_army`` used to be the *only* gate, and that is a deadlock
            # the moment the production stage cannot reach it. Measured at
            # difficulty 1 (``.runtime/combined_ab.json``): with
            # ``army_target``/``attack_at_army`` raised to 18, the army peaked at
            # **17** and ``final_own_army`` was still 17 at the tick cap while the
            # minimum distance to any enemy building was **720.9 px** -- the army
            # never left home. The same economy with the default target of 10 closed
            # to **78.8 px**. So a threshold the economy cannot fund does not make
            # the army stronger, it makes the army passive: the stage waits for a
            # unit the factory never produces and the episode times out with a full
            # bank.
            #
            # Production can miss the target for reasons outside this stage's
            # control (a lost factory, an enemy push, a bank drained by upgrades).
            # The army that exists is still the army that has to fight, so the stage
            # fires anyway once the army has plateaued at a real fraction of the
            # cap.
            # The cap comes from the game config, not ``macro.army_cap``: the rule
            # layer leaves that field unset (None), and ``int(None)`` raised
            # TypeError on every decision, which the dispatcher swallowed as "the
            # stage had nothing to say". Measured on Lake at difficulty 1: the
            # attack stage fired zero orders in 1600 decisions while 15 tanks sat
            # 1250 px from an enemy building and both bases stayed at 100% HP.
            macro_cap = getattr(self.macro, "army_cap", None)
            game_cap = getattr(self.game, "economy_army_cap", None)
            cap = int(macro_cap) if macro_cap is not None else (
                int(game_cap) if game_cap is not None else int(self.macro.attack_at_army)
            )
            cap = max(cap, int(self.macro.attack_at_army))
            # 0.35 of cap is reachable at difficulty 1 (~11 tanks of 32); 0.6
            # (~19) was only reached at step 1420 of 1600, too late to break a
            # building before the tick cap. March with what exists and reinforce.
            if len(ours) < max(1, int(cap * 0.35)):
                return []
        # Address the army group by group, rotating each decision.
        #
        # ``unit_group`` selects members by ``unit.id % groups`` (the bridge's
        # ``addOwnOrderableUnits``), and the broadcast slot (``micro.groups``) means
        # "every living friendly unit" -- which **includes the builder**. Two
        # constraints follow, and both were measured:
        #
        # 1. The broadcast slot cancels the builder's construction, so the attack
        #    stage must not use it. With the broadcast slot firing every decision,
        #    occupied pools collapsed from 2-3 to **1** and the win rate fell from
        #    66.7% to 8-16%.
        # 2. A single fixed group only moves the units that happen to live in it.
        #    Produced units receive incrementing entity ids, so a 10-unit army is
        #    spread across all four groups -- ordering one group means most of the
        #    army never receives an order. Measured symptom
        #    (``.runtime/win_vs_timeout.json``): the army won the field (0-1 enemy
        #    units left) yet **4-5 enemy buildings survived** to the tick cap,
        #    because the units that could have finished them were never ordered to.
        #
        # Rotating covers the whole army within ``groups`` decisions while never
        # touching a group that contains only the builder.
        builder_ids = {
            int(unit.get("entity_id", -1))
            for unit in own
            if unit.get("is_builder")
        }
        groups = max(1, int(self.micro.groups))
        by_group: dict[int, list[int]] = {}
        for unit in ours:
            entity_id = int(unit.get("entity_id", 0))
            if entity_id in builder_ids:
                continue
            group = int(unit.get("group_id") or entity_id % groups)
            by_group.setdefault(group, []).append(entity_id)
        if not by_group:
            return []
        # Deterministic rotation, so two runs on the same seed behave identically.
        ordered_groups = sorted(by_group)
        rotation = int(getattr(self, "_attack_rotation", 0))
        army_group = ordered_groups[rotation % len(ordered_groups)]
        self._attack_rotation = (rotation + 1) % len(ordered_groups)
        foes = state.get("visible_enemy_entities") or []
        # Target the enemy's *capability*, not its centroid.
        #
        # Victory on a stock skirmish map is decided by ``Team.updateDefeatCheck``,
        # which asks whether a team still owns a factory, a builder or a main
        # building -- not whether its units are dead. Marching at the enemy centroid
        # therefore produces exactly the observed stalemate: measured across 12
        # episodes (66.7% win, 0 losses, 4 timeouts), the timeout games ended with
        # 15-16 own units and 10-12 enemy units alive, i.e. the army arrived, fought,
        # killed units, and left the buildings standing.
        #
        # So the target is preferred in this order: command center (the capability
        # anchor) -> any enemy factory -> any enemy building -> enemy centroid.
        def _priority(unit: dict) -> int:
            name = str(unit.get("unit_type") or "").lower()
            if "commandcenter" in name:
                return 0
            if unit.get("is_factory"):
                return 1
            if unit.get("is_building"):
                return 2
            return 3

        enemy_buildings = [
            unit for unit in foes if unit.get("is_building") or unit.get("is_factory")
        ]
        # March at the command center specifically.
        #
        # This ordering was measured the hard way. Targeting the *nearest* enemy
        # building (a defensive turret, usually) parks the army at the enemy's
        # outer ring: occupied pools collapsed from 2-3 to **1** and the win rate
        # fell from 66.7% to 8.3% / 16.7% across 12-episode samples. The command
        # center is the right objective because ``Team.updateDefeatCheck`` treats a
        # team as defeated when it has no factory, builder or main building, and on
        # Lake the command center is the main building -- kill it and the capability
        # check fails.
        command_center = next(
            (
                unit
                for unit in foes
                if "commandcenter" in str(unit.get("unit_type") or "").lower()
            ),
            None,
        )
        target = command_center or (sorted(foes, key=_priority)[0] if foes else None)
        # Aim at the enemy's command center, or the nearest surviving structure.
        #
        # ``attack_move`` walks toward the named point and, on arrival, the units
        # engage whatever is in range. Aiming at the command center is what wins:
        # it is the main building in ``Team.updateDefeatCheck``'s capability test,
        # so destroying it removes the enemy's ability to be "not defeated".
        #
        # Rotating the aim across all surviving buildings was tried and measured
        # **worse** (50% vs 57%): the army wandered between structures instead of
        # concentrating fire, so nothing died quickly. Concentration beats coverage
        # here, and the command center is the single most valuable target.
        if target is None and enemy_buildings:
            cx0 = sum(float(u.get("x") or 0.0) for u in ours) / len(ours)
            cy0 = sum(float(u.get("y") or 0.0) for u in ours) / len(ours)
            target = min(
                enemy_buildings,
                key=lambda unit: (
                    (float(unit.get("x") or 0.0) - cx0) ** 2
                    + (float(unit.get("y") or 0.0) - cy0) ** 2
                ),
            )
        if target is not None:
            tx = float(target.get("x") or 0.0)
            ty = float(target.get("y") or 0.0)
        elif own:
            # Nothing visible: march at the map position furthest from our own
            # centroid, which on a two-player skirmish map is the enemy side.
            own_cx = sum(float(u.get("x") or 0.0) for u in own) / len(own)
            own_cy = sum(float(u.get("y") or 0.0) for u in own) / len(own)
            tx = float(self.micro.map_width) - own_cx
            ty = float(self.micro.map_height) - own_cy
            tx = max(0.0, min(float(self.micro.map_width), tx))
            ty = max(0.0, min(float(self.micro.map_height), ty))
        else:
            return []
        # Distance from the army centroid to the target. While it is still large the
        # army is marching, so re-issuing the order each decision is what keeps it
        # moving in the right direction; once contact is made the policy takes over.
        cx = sum(float(u.get("x") or 0.0) for u in ours) / len(ours)
        cy = sum(float(u.get("y") or 0.0) for u in ours) / len(ours)
        gap = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
        # Keep pressing while the enemy still *has* a capability to destroy.
        #
        # The engage-distance cut-off alone was not enough: measured over 12
        # episodes it left 4 stalemates in which the army stood next to an intact
        # enemy base for the remaining ~9000 ticks. Re-issuing while enemy buildings
        # stand keeps ``attack_move`` aimed at them, which is what converts a won
        # fight into a decided game.
        if gap <= float(self.macro.engage_distance) and not enemy_buildings:
            return []
        # Only re-issue when the order would actually change something: the army
        # centroid moves every decision, so recomputing the grid cell each time is
        # correct (unlike the build stages, whose *target* is a fixed structure).
        gx = int(round(tx / max(1.0, float(self.micro.map_width)) * (self.micro.position_grid - 1)))
        gy = int(round(ty / max(1.0, float(self.micro.map_height)) * (self.micro.position_grid - 1)))
        gx = max(0, min(int(self.micro.position_grid) - 1, gx))
        gy = max(0, min(int(self.micro.position_grid) - 1, gy))
        action = np.array(
            [
                ACTION_KINDS.index("attack_move"),
                int(army_group),
                int(gx),
                int(gy),
                0,
            ],
            dtype=np.int64,
        )
        return self._send_curriculum_order(
            action,
            {"kind": "attack_move", "grid": [int(gx), int(gy)], "drive": True},
        )

    def _issue_late_expansion(self) -> list[dict[str, Any]]:
        """Build another extractor once the factory chain is fully established.

        This is the "learn the economy" half that was missing entirely, and it runs
        **last** so it can never steal the builder from the factory or production
        stages. Measured state of the noop baseline on ``[p2]Lake (2p)`` before this
        stage existed, across every episode:

            peak_own = 5      (commandCenter, builder, extractor, landFactory, 1 unit)
            income   = 26     (from a single T1 extractor, forever)
            funds    = 14182  (unspent, at the tick cap)

        A bank of 14182 that never becomes income is the single clearest statement
        that the environment never asked the agent to invest. The AI, meanwhile,
        ends with 4-5 extractors plus a T2 one.

        Gates, each present because the alternative was measured to be worse:

        * **A finished factory must exist** and its queue must be empty. Without
          this the stage grabs the builder before the factory is even ordered and
          the army never exists -- measured (``.runtime/production_trace.json``):
          ``no_combat_factory`` was the blocking reason on **1968 of 1968**
          decisions, the episode ended with 5 extractors and 19748 unspent credits,
          and zero combat units.
        * **A free pool must exist.** Redirecting at an occupied pool restarts that
          pool's extractor instead of adding capacity.
        * **Credits must clear the reserve**, so expansion never starves the army.
        * **Every standing structure must be finished**, or the builder is mid-build
          and a new order would cancel it.
        """

        if not getattr(self.game, "opening_extractor", False):
            return []
        if not self.macro.allows("expansion"):
            # The commander cancelled expansion: a new extractor is a static
            # building in the open, which is the last thing to buy when the own
            # side cannot defend it.
            return []
        state = self.state
        own = state.get("own_entities") or []
        # The factory chain must be established first: a finished, non-command-centre
        # factory with nothing queued.
        combat_factories = _combat_factories(state)
        if not combat_factories:
            return []
        # NOTE: a busy factory queue deliberately does **not** block expansion.
        #
        # It did, and it was the dominant blocker: measured over 1968 decisions
        # (``.runtime/expansion_trace.json``) ``factory_queue_busy`` was the reason
        # on **1127** of them and expansion only fired on 152, so the economy sat at
        # 2 extractors for the whole episode while the factory steadily consumed the
        # income. A queueing factory does not touch the builder -- production and
        # construction are different units doing different jobs -- so waiting for it
        # to go idle was waiting for something that never happens.
        slots = state.get("resource_slots") or []
        free = [
            index
            for index, slot in enumerate(slots)
            if isinstance(slot, dict) and not slot.get("occupied")
        ]
        if not free:
            return []
        # Every structure currently standing must be finished, or the builder is
        # busy somewhere and a new order would cancel that work.
        #
        # **Buildings only, and only ones the builder is actively working on.**
        # The original form of this check was
        # ``any(not _is_operational(unit) for unit in own if unit.get("is_building"))``
        # which is subtly wider than "the builder is busy": on a real economy map
        # the enemy destroys a factory every few hundred ticks, the factory stage
        # immediately re-orders it, and the reconstruction keeps *some* building
        # permanently below 1.0 progress. Expansion then never runs -- measured on
        # Lake at difficulty 1 (``.runtime/army_growth.json``): ``income`` sat at
        # exactly **26.0** (one extractor) for 14,000 ticks while ``factories``
        # oscillated 4 -> 3 -> 0 -> 3, i.e. the army was losing its production
        # buildings faster than the builder could replace them and the economy could
        # never grow to pay for replacements.
        #
        # A structure at progress < 1.0 that the builder is *not* currently assigned
        # to is not a reason to refuse an expansion order; the builder would simply
        # start the extractor. What must be avoided is stealing the builder from an
        # in-flight build, and that is what the ``_builder_busy_hold`` mechanism
        # already handles in the factory stage.
        if any(
            not _is_operational(unit)
            for unit in own
            if unit.get("is_building")
            and isinstance(unit.get("build_progress"), (int, float))
            and 0.0 < float(unit.get("build_progress")) < 1.0
        ):
            return []
        if not any(
            unit.get("is_builder") and float(unit.get("hp") or 0.0) > 0.0 for unit in own
        ):
            return []
        funds = float((state.get("global_state") or {}).get("funds") or 0.0)
        # An extractor is a *production investment*, so it competes with the army
        # for the same credits -- but the reserve gate must not be the price of a
        # structure, or nothing ever gets built.
        #
        # The original gate was ``funds < macro.credit_reserve`` (900 by default,
        # derived from the cheapest buildable structure). Measured funds on Lake at
        # difficulty 1 wobble between 216 and 888 while the factory stage keeps
        # spending every credit above the reserve, so the gate was closed on most
        # decisions and expansion -- the only stage that raises the income ceiling --
        # never fired.
        #
        # The extractor's own price is the honest bar: if the bank can pay for the
        # extractor, it can be ordered. ``credit_reserve`` still protects the
        # *builder's next structure* via the factory stage, which is where that
        # concern belongs.
        #
        # This build does not expose a per-menu price array (``reported_price``
        # arrived in a later commit), so the engine's own reported price is read
        # from the builder's menu when available and otherwise the configured
        # ``extractor_cost`` is used, which is the same number the action mask
        # gates on.
        extractor_price = float(getattr(self.game, "extractor_cost", 700.0) or 700.0)
        for entry in state.get("builders") or []:
            if not isinstance(entry, dict):
                continue
            names = entry.get("can_build")
            prices = entry.get("can_build_price")
            if not isinstance(names, list) or not isinstance(prices, list):
                continue
            for name, price in zip(names, prices):
                if str(name).lower() in {"extractor", "extractort1"}:
                    try:
                        extractor_price = float(price)
                    except (TypeError, ValueError):
                        pass
                    break
        if funds < extractor_price:
            return []
        occupied = sum(
            1 for slot in slots if isinstance(slot, dict) and slot.get("occupied")
        )
        if occupied >= int(self.macro.max_extractors):
            return []
        # Rotate through the free pools instead of always aiming at the first one.
        #
        # A pool whose extractor the engine silently drops never becomes "occupied",
        # so an always-first policy keeps re-ordering the same dead pool forever and
        # income never moves. Measured on Lake at difficulty 1 with 9 pools visible and
        # ``economy_max_extractors = 6``: ``build_extractor`` was dispatched **33
        # times** and the occupied count stayed at **1 for 1991 of 2006 decisions**,
        # with 8 pools free the entire episode and income pinned at 26.0
        # (``.runtime/income_breakdown.json``). Seven pools' worth of income -- roughly
        # +50 credits/s -- was never attempted twice.
        #
        # The rotation offset advances only when a pool is *proven* dead (see the
        # distance gate below), never merely because time passed.
        #
        # The target is **latched** for the whole leg. Recomputing ``free[offset]``
        # every decision looks equivalent but is not: ``free`` is rebuilt from the
        # current occupancy, so the moment a pool's state changes the ordinal points
        # at a different pool and the builder is re-tasked mid-walk. That is the same
        # failure as advancing the offset too early, just via the other variable.
        latched = getattr(self, "_expansion_target", None)
        if latched is not None and int(latched) in free:
            chosen_pool = int(latched)
        else:
            offset = int(getattr(self, "_expansion_pool_offset", 0)) % max(1, len(free))
            chosen_pool = int(free[offset])
            self._expansion_target = chosen_pool
            # A new target is a new leg: its distance history and its time budget
            # both start from here.
            self._expansion_distance = None
            self._expansion_ordered_at = None
            # The leg's *starting* distance is what the stall budget must be sized
            # against; see the ``_expansion_walk_budget`` call below for the
            # measurement that makes this load-bearing.
            self._expansion_leg_distance = None
        action = np.array(
            [
                ACTION_KINDS.index("build_extractor"),
                int(self.micro.groups),
                int(self.micro.position_grid) // 2,
                int(self.micro.position_grid) // 2,
                chosen_pool,
            ],
            dtype=np.int64,
        )
        # Re-issue only when the previous order actually **stalled**, measured by
        # whether the builder is still *closing the distance* to the pool.
        #
        # This stage used to re-order unconditionally whenever its gates opened.
        # Measured in the real game (``.runtime/macro_ab.json``, 366 decisions, noop
        # policy): **194** ``build_extractor`` orders went out, i.e. one every other
        # decision. Each one restarts the construction it is trying to finish
        # (``.runtime/ab_reissue.py``: ``build_progress`` pinned at 0.0 forever while
        # every receipt says ``accepted: true``), so the economy stopped growing
        # while every receipt claimed success.
        #
        # A progress gate (the one the opening-extractor stage uses) is **not
        # enough here**, and the reason is specific to this stage: it aims at a
        # *free pool*, and a pool the builder has been sent to but whose hull the
        # bridge has not streamed yet still reads as free. There is therefore no
        # half-built entity to watch.
        #
        # The gate used to be "the occupied count has not changed for
        # ``STALL_DECISIONS`` decisions". That conflates **four** states which need
        # opposite responses, and the conflation is what kept the economy at one
        # extractor:
        #
        #   * the builder is still walking there      -> keep waiting
        #   * it arrived and is building              -> keep waiting
        #   * it arrived and the engine refuses       -> switch pools
        #   * the order never reached the engine      -> fix the order, not the pool
        #
        # Measured (``.runtime/extractor_pools_probe.json``, 600 ticks per pool):
        # the walk to the nearest *free* pool is 1147 ticks at the builder's
        # 0.8 px/tick, but the window was 20 decisions x 10 ticks = **200 ticks**
        # -- 3.8x too short. So the stage advanced to the next pool while the
        # builder was still en route, the builder turned around, and it never
        # arrived anywhere. Nine pools, 47 orders, one extractor.
        #
        # The observable that separates the four is the **distance from the builder
        # to the pool it was sent to** (both are already in the payload). While that
        # shrinks, the order is working and nothing should be re-issued.
        builder = next(
            (
                unit for unit in own
                if unit.get("is_builder") and float(unit.get("hp") or 0.0) > 0.0
            ),
            None,
        )
        target_slot = next(
            (
                slot for slot in slots
                if isinstance(slot, dict) and int(slot.get("slot_id", -1)) == chosen_pool
            ),
            None,
        )
        distance_now: float | None = None
        if builder is not None and target_slot is not None:
            try:
                distance_now = float(
                    np.hypot(
                        float(builder.get("x") or 0.0) - float(target_slot.get("x") or 0.0),
                        float(builder.get("y") or 0.0) - float(target_slot.get("y") or 0.0),
                    )
                )
            except (TypeError, ValueError):
                distance_now = None
        # The builder walks at its own speed, so "closing" is judged against the
        # distance it could cover in one decision (``move_speed`` px/tick x the
        # bridge's 10-tick decision interval), with the threshold halved so a
        # diagonal path still counts as progress.
        speed = 0.8
        if builder is not None:
            try:
                reported = float(builder.get("move_speed") or 0.0)
            except (TypeError, ValueError):
                reported = 0.0
            if reported > 0.0:
                speed = reported
        # ``closing_threshold`` is the smallest distance change that counts as
        # "still walking". It used to be ``speed * 10 * 0.5`` = 4 px for the
        # builder's 0.8 px/tick over the bridge's 10-tick decision interval, i.e.
        # half a step. A builder that walks on a diagonal, is nudged by the engine's
        # own pathing, or pauses for one decision therefore reads as "not closing",
        # and the stage treats a working order as a stalled one.
        #
        # Measured (``.runtime/expansion_timeline.json`` on Lake at difficulty 1):
        # the builder closed 894 -> 692 px on one leg and the stage still switched
        # pools, ending on one 1663 px away. In 1400 decisions it never arrived
        # anywhere and income never left 26.
        #
        # One full step is the honest bar for "made progress this decision".
        closing_threshold = max(1.0, speed * 10.0)
        # Record the distance this leg STARTED from. The stall budget is sized from
        # it, and it must not shrink as the builder approaches -- see the
        # ``_expansion_walk_budget`` call below.
        if distance_now is not None and getattr(self, "_expansion_leg_distance", None) is None:
            self._expansion_leg_distance = float(distance_now)
        distance_then = getattr(self, "_expansion_distance", None)
        if (
            distance_now is not None
            and distance_then is not None
            and distance_now < float(distance_then) - closing_threshold
        ):
            # Still walking: the order is working. Record the distance and get out
            # of the way without touching the stall counter.
            self._expansion_distance = distance_now
            self._expansion_stalled = 0
            # ``_expansion_ordered_at`` is deliberately NOT refreshed here: it
            # anchors "how long has this leg been running", and rewriting it on
            # every closing decision makes the stall budget unreachable.
            self._expansion_occupied = int(occupied)
            return []
        self._expansion_distance = distance_now
        occupied_now = occupied
        ordered_at = getattr(self, "_expansion_ordered_at", None)
        occupied_then = getattr(self, "_expansion_occupied", None)
        progress = False
        if occupied_then is not None and occupied_now != occupied_then:
            # A new extractor appeared: this is a fresh, legitimate expansion.
            # Release the latch so the next leg picks a new pool, and restart the
            # time budget rather than inheriting the previous leg's.
            self._expansion_ordered_at = None
            self._expansion_target = None
            self._expansion_distance = None
            self._expansion_stalled = 0
            progress = True
        if not progress and ordered_at is not None:
            # The budget is the time this leg actually needs, not a constant:
            # ``distance / speed`` plus the structure's own build time. A constant
            # window either cuts off a legal walk (what happened) or delays the
            # switch past the point where the pool is provably dead.
            #
            # **Measured against the distance at dispatch, not the current one.**
            # Using ``distance_now`` made the budget shrink as the builder
            # approached, so the closer it got the sooner it was declared stalled:
            # measured on Lake at difficulty 1 (``.runtime/expansion_timeline.json``),
            # the builder closed from 894 px to 692 px on one leg -- real progress --
            # and the budget computed from the *remaining* 692 px then expired, so
            # the stage switched to a pool 1183 px away, and from there to one
            # 1663 px away. In 1400 decisions it never arrived anywhere: nine free
            # pools, 23 orders, one extractor, income pinned at 26.0.
            #
            # The leg's own length is the honest unit of patience: a leg that starts
            # 900 px away is allowed the time 900 px needs, regardless of how far
            # along it is when the check runs.
            leg_distance = getattr(self, "_expansion_leg_distance", None)
            budget = self._expansion_walk_budget(
                leg_distance if leg_distance is not None else distance_now
            )
            if self.episode_steps - int(ordered_at) < budget:
                return []
            # The budget elapsed without the builder closing the distance. The pool
            # we aimed at is not going to produce an extractor, so advance to the
            # next free pool instead of re-ordering the same dead one.
            #
            # Measured on Lake at difficulty 1, 9 pools visible and
            # ``economy_max_extractors = 6``: ``build_extractor`` was dispatched
            # **33 times** while the occupied count stayed at **1 for 1991 of 2006
            # decisions** with 8 pools free the whole episode and income pinned at
            # 26.0 (``.runtime/income_breakdown.json``).
            self._expansion_pool_offset = int(
                getattr(self, "_expansion_pool_offset", 0)
            ) + 1
            self._expansion_target = None
            self._expansion_distance = None
            # This pool is abandoned, so the next leg gets its own starting
            # distance. Leaving the old value here would size the *next* leg's
            # patience from the *previous* leg's length.
            self._expansion_leg_distance = None
        self._expansion_ordered_at = int(self.episode_steps)
        self._expansion_occupied = int(occupied_now)
        return self._send_curriculum_order(
            action,
            {"kind": "build_extractor", "slot_id": chosen_pool, "expansion": True},
        )

    def _expansion_walk_budget(self, distance: float | None) -> int:
        """Decisions to allow before declaring a pool unreachable.

        Derived from the leg the order actually asks for, because a constant was
        wrong by 3.8x: the nearest *free* pool on Lake is 918 px from the builder,
        i.e. 1147 ticks at 0.8 px/tick, while the constant window was 200 ticks.
        The stage therefore switched pools mid-walk, the builder turned around,
        and no second extractor was ever finished.

        Components:

        * **walk** -- ``distance / (move_speed * 10 ticks per decision)``;
        * **build** -- the structure's own construction time. 140 ticks is the
          measured extractor build under no interruption, i.e. 14 decisions;
        * **slack** -- 25%, because the path is not straight, the builder may be
          re-tasked once by the policy, and ``occupied`` only flips when the hull
          is streamed.

        Floored at the old constant so a builder that is already standing on the
        pool keeps the historical (short) patience.
        """

        build_decisions = 14
        if distance is None:
            return max(STALL_DECISIONS, build_decisions)
        speed = 0.8  # builder ``move_speed`` px/tick, measured (.runtime/contract_gap.json)
        walk_decisions = float(distance) / max(1e-6, speed * 10.0)
        return max(
            STALL_DECISIONS,
            int((walk_decisions + build_decisions) * 1.25) + 1,
        )

    def _issue_opening_production(self) -> list[dict[str, Any]]:
        """Keep queueing combat units while a finished factory exists (curriculum).

        Companion of :meth:`_issue_opening_extractor`, and it exists for the same
        reason: the "build a factory, then use it" chain is too long for random
        exploration. Measured on ``[p2]Lake (2p)`` a factory only finishes around
        tick 491 and the first unit around tick 2021, while a from-scratch policy
        spends most decisions on ``build_extractor``; over a full 200-update run
        ``bonus_produce`` was exactly 0.000 in **199 of 199 updates** and the
        agent never fielded a single combat unit.

        **This used to fire exactly once per episode.** That made the whole
        economy a one-shot, and the measurement is unambiguous: the noop baseline
        reached ``peak_own = 5`` (command center, builder, extractor, factory, one
        combat unit) and then stopped, while the bank climbed to **14182 unspent
        credits** and the AI grew to 15 units. A single unit is not an army, so
        the policy's reward for production was one ``bonus_produce`` and then
        nothing -- there was no gradient pointing at "build many".

        The stage now sustains a target army size, which is what makes "learn the
        economy" mean anything: the factory keeps producing while credits allow.

        Two guards are load-bearing, both measured:

        * **A finished factory must exist** -- an under-construction one
          advertises an empty menu and every early order came back
          ``PRODUCE_UNAVAILABLE``, which taught the policy to avoid the action.
        * **It must not starve the builder.** Ordering a unit every single
          decision drains the bank to ~496 credits *and cancels the in-progress
          extractor*, because every real order interrupts the builder. Measured
          (``.runtime/produce_spam.json``): spamming production for one episode
          ended with ``builder x25``, ``income`` down from 26 to 18, the extractor
          gone, and still **zero** combat units in the field -- the command center
          was the only surviving "combat" entity. Production is therefore gated on
          the bank staying above a reserve that is strictly larger than one unit.

        This removes the exploration barrier only. The order goes through the
        normal bridge path, the engine may still refuse it, and ``bonus_produce``
        is still paid by the observation stream when a *new combat unit appears* --
        so the reward stays honest and cannot be farmed by repeating the order.
        """

        if not getattr(self.game, "opening_production", False):
            return []
        if not self.macro.allows("production"):
            # The commander cancelled production for this decision. Unlike the
            # build kinds this is *never* a way to protect the builder (production
            # addresses a factory, not the builder), so the only reason to set it
            # is a plan that wants the bank for something else entirely.
            return []
        state = self.state
        own = state.get("own_entities", [])
        # Only a *finished* factory can queue: an under-construction one advertises
        # an empty menu (see ``IronZeroBridge.factoriesJson``) and every early order
        # came back ``PRODUCE_UNAVAILABLE``, which taught the policy to avoid the
        # action entirely.
        #
        # The command centre reports ``is_factory`` too (it builds builders), but a
        # finished command centre exists from tick 1 and its only menu entry is
        # ``builder`` -- queueing there would pay nothing and teach nothing. The
        # bridge already answers the real question through ``factories[].can_produce``,
        # so require a factory that advertises a non-builder entry.
        combat_factories = _combat_factories(state)
        if not combat_factories:
            return []
        funds = float((state.get("global_state") or {}).get("funds") or 0.0)
        # Raise extra builders first, before any combat unit.
        #
        # Measured on Lake at difficulty 1 (round 17): the enemy reaches 2-3
        # builders by tick 9011 while we are pinned at exactly 1, because
        # ``_UNIT_COMBAT_RANK`` scores ``builder`` as 0 and the selection loop
        # below therefore can never choose it. Every structure -- extractor,
        # factory, turret -- then queues behind one unit walking the map, so
        # expansion and production serialise against each other.
        #
        # This is deliberately the *first* thing the stage does once a factory
        # exists: a second builder is what lets an extractor and a factory be
        # built at the same time, which is the difference between an economy that
        # grows and one that crawls. It is capped so it cannot become a sink.
        builder_quota = int(getattr(self.game, "economy_max_builders", 0) or 0)
        if builder_quota > 0:
            live_builders = sum(
                1
                for unit in own
                if unit.get("is_builder") and float(unit.get("hp") or 0.0) > 0.0
            )
            queued_builders = sum(
                int(entry.get("queue_length") or 0)
                for entry in combat_factories
                if "builder" in [str(n) for n in (entry.get("can_produce") or [])]
            )
            if live_builders < builder_quota and not queued_builders:
                for entry in combat_factories:
                    menu = [str(name) for name in (entry.get("can_produce") or [])]
                    if "builder" not in menu:
                        continue
                    if entry.get("is_built") is False:
                        continue
                    prices = entry.get("can_produce_price") or []
                    position = menu.index("builder")
                    price = None
                    if position < len(prices):
                        try:
                            price = float(prices[position])
                        except (TypeError, ValueError):
                            price = None
                    if price is None:
                        price = float(_UNIT_PRICE.get("builder", 500))
                    if price > funds:
                        continue
                    factory_id = entry.get("entity_id")
                    if factory_id is None:
                        continue
                    orders = self._send_direct_order(
                        {"kind": "produce_unit", "factory_entity_id": int(factory_id),
                         "unit_type": "builder"},
                        {"kind": "produce_unit", "factory_entity_id": int(factory_id),
                         "unit_type": "builder"},
                    )
                    if orders:
                        return orders
        # Keep an army, not a token: stop once the target is reached so the stage
        # does not spend the whole bank on units the map cannot use.
        combat_alive = sum(
            1
            for unit in own
            if unit.get("can_attack")
            and not unit.get("is_builder")
            and not unit.get("is_building")
            and float(unit.get("hp") or 0.0) > 0.0
        )
        if combat_alive >= int(self.macro.army_target):
            self.production_curriculum_done = True
            # Past the army target the factory may keep replacing losses, up to
            # ``economy_army_cap``. Measured: the remaining stalemates all ended
            # with ``peak_own >= 17`` against 13-15 surviving AI units, so stopping
            # production at the target let the AI out-reinforce us in the endgame.
            if combat_alive >= int(self.macro.army_cap):
                return []
        else:
            self.production_curriculum_done = False
        # Do not spend the builder's budget. ``economy_credit_reserve`` is larger
        # than one cheap unit, so the extractor/factory chain can always finish.
        # (``funds`` was read above, before the builder top-up, so the two paths
        # cannot disagree about the bank within one decision.)
        # While an upgrade is still owed, hold its price back as well.
        #
        # Measured on Lake at difficulty 1 (scripts/probe_upgrades.py): with the
        # upgrade stage finally reachable, it still emitted its four orders at
        # tick ~17000 of a 19674-tick episode and no building ever left
        # tech_level 1. The reason is here rather than there -- this stage drains
        # the bank to ``credit_reserve`` (700) on every decision, so the upgrade
        # stage's own 1200 gate only opens once the army has saturated and the
        # bank drifts up on its own. An upgrade ordered 2600 ticks before the cap
        # cannot repay itself, so "late but eventually ordered" is the same as
        # never. Raising the floor here is what lets it start in the opening.
        # ...but only once there is an army to show for it.
        #
        # The upgrade stage requires a working factory AND at least one combat
        # unit, so saving from decision 1 deadlocks: no army is built (the floor
        # blocks this stage), so the upgrade gate never opens. Measured with
        # credit_reserve=2600 (round 18): funds climbed to 3288 with zero upgrade
        # orders because ``combat_alive`` stayed 0 all episode.
        #
        # The honest sequence is therefore two-phase: build the first combat unit
        # at the plain reserve, then save the upgrade's price on top of it.
        upgrade_floor = 0.0
        if (
            getattr(self.game, "economy_upgrade", False)
            and int(self._upgrades_issued) < int(self.macro.max_upgrades)
            and int(combat_alive) >= 1
        ):
            upgrade_floor = float(self.micro.extractor_upgrade_cost)
        if funds < float(self.macro.credit_reserve) + upgrade_floor:
            return []
        # While extractors can still be added, every unit must leave the bank able
        # to pay for one more extractor. Expansion is the ONLY stage that raises
        # income, and this stage competes with it for the same credits.
        #
        # Measured on Lake at difficulty 1 with ``credit_reserve = 700``
        # (scripts/probe_hover_gates.py): two otherwise identical runs differed
        # only in which unit production preferred --
        #
        #   hoverTank (450 cr, value/credit 6/450) -> funds<700 on 1723 decisions,
        #     expansion fired ONCE, extractors 1, income 26
        #   tank      (350 cr, value/credit 4/350) -> funds<700 on 50 decisions,
        #     expansion fired 4 times, extractors 3, income 42
        #
        # The expensive unit drained the bank below the extractor price, the
        # expansion gate stayed shut, income never rose, and funds recovered even
        # more slowly -- a self-reinforcing stall where picking the better combat
        # unit made the whole economy worse. The protection belongs on the *unit
        # price* (checked per candidate below, where the price is known) rather
        # than as a flat addend on this gate: adding ``extractor_cost`` here made
        # the bar 1400 and shrank the army (peak 17-22 -> 13-16) while buying no
        # extra wins.
        protect = _occupied_extractors(state) < int(self.macro.max_extractors)
        extractor_price = float(getattr(self.game, "extractor_cost", 700.0) or 700.0)
        # Top up the queue instead of going silent whenever it is non-empty.
        #
        # This used to return ``[]`` as soon as *any* factory had
        # ``queue_length > 0``, on the theory that re-ordering would restart
        # production the way re-issuing a build order restarts construction. The two
        # are not alike, and the cost was measured: on Lake at difficulty 1 the
        # curriculum chose ``produce_unit`` on only **9 of 2100 decisions** while the
        # factories stood intact the whole time and the queue was busy on 540 of the
        # 836 decisions after the army peaked. The army therefore fell from 11 units
        # to 5 against an enemy that grew 15 -> 31, with 600+ unspent credits in the
        # bank (``.runtime/army_decline.json``).
        #
        # Rusted Warfare's factory queue is a *pipeline*, not a one-shot task: the
        # queue drains as units are delivered, so a non-empty queue is the normal
        # state of a factory that is working. What must be avoided is spamming a
        # *new* order every decision (``.runtime/produce_spam.json``: that drained
        # the bank to ~496 and cancelled the in-progress extractor). The distinction
        # is headroom, not emptiness: keep the queue topped up to a small number of
        # pending units and let it drain.
        queued = sum(
            int(entry.get("queue_length") or 0) for entry in combat_factories
        )
        if queued >= int(self.macro.production_queue_depth):
            return []
        # Name the factory and the unit explicitly; never route through an index.
        #
        # The index form was the source of two separate wedges. First the target head
        # was pinned at ``target_slots - 1``, which resolved modulo a list covering
        # every factory including the command centre, so a land factory was ordered
        # to build ``scout`` -- absent from its menu -- and the queue item sat at
        # progress -1.0 from tick 1001 to 2751. Then the index was chosen from
        # ``produce_options``, which is rebuilt from the live observation and *shrinks*
        # whenever a factory is busy: measured with scripts/probe_produce_options.py,
        # 13 entries at step 83 and 6 at step 85, so a fixed index names a different
        # pair one decision later and the queue ended up holding ``c_tank``.
        #
        # The bridge accepts the concrete form directly
        # (``action.getLong("factory_entity_id")`` + ``action.getString("unit_type")``),
        # so there is no reason to express a decision the curriculum already knows.
        funds = float((state.get("global_state") or {}).get("funds") or 0.0)
        chosen_factory = None
        chosen_unit = None
        best_score = None
        for entry in combat_factories:
            factory_id = entry.get("entity_id")
            if factory_id is None:
                continue
            # Only a factory the engine can actually work on may receive orders.
            # ``Factory.update`` returns while isBuiltAndNotTransported() is false,
            # so a queue item at an unfinished factory sits at progress -1.0
            # forever and holds every ``production_queue_depth`` slot. Measured on
            # Lake at difficulty 1: three tanks queued at factory 27 with
            # buildProgress=0.0 and the army stayed at zero for 19000 ticks.
            if entry.get("is_built") is False:
                continue
            menu = [str(name) for name in (entry.get("can_produce") or [])]
            prices = entry.get("can_produce_price") or []
            for position, name in enumerate(menu):
                price = None
                if position < len(prices):
                    try:
                        price = float(prices[position])
                    except (TypeError, ValueError):
                        price = None
                if price is None:
                    price = float(_UNIT_PRICE.get(name.split(".")[-1], 500))
                # Affordability is part of the decision: a 900-credit artillery that
                # the bank cannot pay for is worth less than a 350-credit tank that
                # gets built. Ordering the unaffordable one measured as an *empty*
                # army over a full episode.
                if price > funds:
                    continue
                # ...and it must leave room for the economy to keep growing, while
                # extractors remain (see ``protect`` above). This is the exact form
                # of that rule: spend ``price``, keep ``extractor_price``.
                if protect and funds - price < extractor_price:
                    continue
                score = _unit_value_per_credit(name)
                if score <= 0.0:
                    continue
                if best_score is None or score > best_score:
                    best_score = score
                    chosen_factory = int(factory_id)
                    chosen_unit = name
        if chosen_factory is None or chosen_unit is None:
            # Nothing buildable and affordable this decision. Returning here is
            # correct: ordering anything else wedges the queue (see above).
            return []
        orders = self._send_direct_order(
            {
                "kind": "produce_unit",
                "factory_entity_id": int(chosen_factory),
                "unit_type": str(chosen_unit),
            },
            {"kind": "produce_unit", "factory_entity_id": int(chosen_factory),
             "unit_type": str(chosen_unit)},
        )
        if not orders:
            return []
        self.production_curriculum_done = False
        return orders

    def _issue_opening_factory(self) -> list[dict[str, Any]]:
        """Queue one land factory once the opening extractor is finished.

        This is the link that was missing between :meth:`_issue_opening_extractor`
        and :meth:`_issue_opening_production`, and its absence is why
        ``bonus_produce`` was structurally unreachable.

        Measured on ``[p2]Lake (2p)`` with a pure ``noop`` policy (so no order of
        ours can cancel any construction) over a full 20056-tick episode:

            extractor finished   tick 151
            non-CC factory       never  (still only commandCenter + extractorT1
                                          at tick 20056)
            bonus_produce        0.0

        The extractor curriculum builds the first structure and the production
        curriculum requires a *finished factory*, but nothing ordered the factory
        itself. ``_issue_opening_production`` therefore never fired: both 40-update
        PPO calibration runs reported ``bonus_produce`` = 0 in **40/40** updates
        while ``bonus_build`` was nonzero in 17/40 -- the agent learned extractors
        and stopped.

        The gap is not position legality: a scan of all 64 cells of the policy's
        position head was accepted **64/64**. It is not interruption either:
        ordering the factory once and then only nooping finishes it at tick 1891,
        and re-issuing it every decision also finishes (tick 2611). Nothing in the
        curriculum ever ordered it.

        Returns the receipt-ish dict, or ``None`` when the curriculum is off, the
        extractor is still building, a non-command-centre factory already exists,
        or there is no builder -- all ordinary situations, so this never raises.
        """

        if not getattr(self.game, "opening_factory", False):
            return []
        if not self.macro.allows("factory"):
            # The commander cancelled the factory chain. The stage is a no-op then,
            # and the production stage's own "a finished factory must exist" gate
            # keeps the two consistent: no factory means no units.
            return []
        state = self.state
        own = state.get("own_entities", [])
        # Re-issue until the factory is **finished**, not merely started.
        #
        # Both halves were measured and both are load-bearing, exactly as for the
        # extractor stage:
        #
        # * stopping as soon as a ``landFactory`` entity existed handed a
        #   half-built factory back to the policy, which cancelled it, and nothing
        #   ever restarted it. Masked-random Lake: the factory appeared at tick
        #   7811 and was still at ``build_progress < 1`` when the 20056-tick episode
        #   ended -- ``bonus_produce`` 0.0, ``production_curriculum_done`` false,
        #   so stage 3 never fired over 2006 decisions;
        # * stopping only on a "sent once" latch (before the entity check existed)
        #   let the policy cancel the *only* order, and the stage never came back.
        factories = [
            unit for unit in own
            if str(unit.get("unit_type") or "").lower().endswith("factory")
        ]
        if any(_is_operational(unit) for unit in factories):
            # One finished factory satisfies "the chain works", but more factories
            # raise the reinforcement rate, so fall through to the count check
            # below instead of returning here.
            self.factory_curriculum_done = True
        # The quota is a plain count of operational land factories.
        #
        # This replaced a tangled pair of branches that produced an infinite re-order
        # loop. The old code cleared ``factory_site`` whenever a factory was
        # operational but the count was below the quota, and returned ``[]`` only in
        # an ``elif`` -- so with one factory standing and the quota at two, it
        # re-latched the site, ordered, cleared, re-latched, and never stopped.
        # Measured (``.runtime/dispatch_tap.json``): ``build_structure:landFactory``
        # was dispatched **1736 times in a single episode**; the builder never finished
        # anything else, the army collapsed to one ``scout``, and 4/4 episodes timed
        # out -- against 4/4 wins before that change.
        #
        # Two questions must stay separate:
        #   * "is the chain established?" -> at least one operational land factory;
        #   * "is the quota met?"          -> ``economy_max_factories`` operational.
        # Only when the quota is met does this stage stop. A *destroyed* factory drops
        # the count, which re-opens the stage, which is the intended repair behaviour.
        land_factories = [
            unit for unit in factories
            if str(unit.get("unit_type") or "").lower() == "landfactory"
        ]
        operational = [
            unit for unit in land_factories if _is_operational(unit)
        ]
        if operational:
            self.factory_curriculum_done = True
        if len(operational) >= int(self.macro.max_factories):
            # Quota met and every one of them finished: nothing left to order.
            return []
        # Re-latch the site **before** deciding anything about in-flight hulls.
        #
        # Ordering is the whole bug. When these two steps were the other way round,
        # clearing the site happened *after* the ``unfinished`` early return, so on the
        # second factory the sequence was: quota not met -> no unfinished hull -> clear
        # the site -> ... and then the site-selection block below ran with
        # ``factory_site`` already cleared, re-computed a site near the builder (which
        # stands at factory #1), and the resulting order was refused. Measured
        # (``.runtime/trained_behaviour.json``): ``structure_types`` contained no
        # ``landFactory`` at all -- only the command centre and extractors -- so the
        # agent never had a ground factory, and every episode fielded a single
        # ``scout``.
        #
        # This clears the **real** attribute, not ``_factory_site``. The two names look
        # like the same field and are not: ``_factory_site`` absorbed the write,
        # ``factory_site`` kept pointing at the *first* factory, and the second factory
        # was therefore ordered at coordinates the first one already occupied -- so
        # ``economy_max_factories = 2`` silently stayed at 1, which is exactly the
        # reported "only builds one factory".
        if operational and self.factory_site is not None:
            self.factory_site = None
            self._factory_progress = None
            self._factory_stalled = 0
        # A half-built hull is the *stall* case, not a reason to wait forever.
        #
        # This block used to read ``if unfinished: return []`` with the rationale
        # that re-issuing restarts construction (``.runtime/ab_reissue.py``). But
        # the stall detector further down -- the one that exists precisely to
        # re-issue after :data:`STALL_DECISIONS` of no progress -- was unreachable
        # behind it, and the site-clear above wiped ``_factory_stalled`` on every
        # call. The second landFactory therefore sat at ``build_progress 0.0`` for
        # the whole episode (``scripts/probe_factory2.py``, tick 2011 -> 19011)
        # and production ran on one factory, halving throughput.
        #
        # Measured: the stage sent 33 orders and then idled 1919 decisions with
        # ``_factory_progress=None`` -- the counter could never accumulate.
        unfinished = [unit for unit in land_factories if not _is_operational(unit)]
        if unfinished:
            target = unfinished[0]
            try:
                progress_now = float(target.get("build_progress") or 0.0)
            except (TypeError, ValueError):
                progress_now = None
            previous = getattr(self, "_factory_progress", None)
            if progress_now is not None and (
                previous is None or progress_now > float(previous) + 1e-9
            ):
                # Real progress: keep waiting and remember where it got to.
                self._factory_progress = progress_now
                self._factory_stalled = 0
                return []
            self._factory_stalled = int(getattr(self, "_factory_stalled", 0)) + 1
            if self._factory_stalled < STALL_DECISIONS:
                return []
            # Stalled at the same progress for a whole stall window. Re-issue at
            # the hull's own coordinates -- never a fresh site, or a third hull
            # would be started next to the two we already have.
            self._factory_stalled = 0
            bx = float(target.get("x") or 0.0)
            by = float(target.get("y") or 0.0)
            self.factory_exact = (bx, by)
            grid = int(self.micro.position_grid)
            self.factory_site = (
                max(0, min(grid - 1, int(round(bx / max(1.0, float(self.micro.map_width)) * (grid - 1))))),
                max(0, min(grid - 1, int(round(by / max(1.0, float(self.micro.map_height)) * (grid - 1))))),
            )
            gx, gy = self.factory_site
            menu = build_menu(self.obs_state)
            action = np.array(
                [ACTION_KINDS.index("build_structure"), int(self.micro.groups),
                 int(gx), int(gy), 0],
                dtype=np.int64,
            )
            if menu:
                action[4] = int(menu_index(menu, "land_factory"))
            return self._send_curriculum_order(
                action,
                {"kind": "build_structure", "structure_type": "land_factory",
                 "target": self.factory_exact},
            )
        # Wait for the opening extractor. Issuing the factory while the builder is
        # mid-construction would *cancel* the extractor -- every order cancels the
        # in-flight one, which is the whole reason ``noop`` exists. A missing
        # ``build_progress`` is treated as finished so an older bridge still works.
        extractors = [
            unit for unit in own if "xtractor" in str(unit.get("unit_type") or "")
        ]
        if not extractors:
            return []
        if any(not _is_operational(unit) for unit in extractors):
            return []
        builder = next(
            (unit for unit in own if unit.get("is_builder") and float(unit.get("hp") or 0.0) > 0.0),
            None,
        )
        if builder is None:
            return []
        # A half-built factory already has a site; rebuild *there* so re-issuing
        # does not restart it somewhere else (which would abandon the progress and
        # leave two half-built hulls).
        #
        # Only an **unfinished** factory pins the site. A finished one must not:
        # its hull is where the *previous* factory was built, so targeting it would
        # ask the engine to place the second factory inside the first. That was the
        # second half of "only ever one factory" -- and it survived the
        # ``_factory_site`` typo above because it re-derived the coordinate from
        # ``factories[0]`` on every call.
        unfinished = [unit for unit in factories if not _is_operational(unit)]
        if unfinished:
            target = unfinished[0]
            bx = float(target.get("x") or 0.0)
            by = float(target.get("y") or 0.0)
        else:
            bx = float(builder.get("x") or 0.0)
            by = float(builder.get("y") or 0.0)
        # Anchor for the exact-coordinate candidates below: our **own base**, not the
        # builder.
        #
        # Anchoring on the builder looks equivalent and is not. By the time this stage
        # runs, the builder has already walked to the opening resource pool (measured:
        # builder at ``(1170, 2050)`` while the command centre sits at ``(990, 2010)``),
        # and offsets from *that* point fall into the lake -- measured
        # (``.runtime/build_receipts.json``) every factory order went to
        # ``(1117.71, 1775.43)``, which the engine silently drops while still reporting
        # ``accepted``. The base centre is land by construction and is where a second
        # factory belongs anyway.
        base_anchor = next(
            (unit for unit in own if unit.get("is_factory_building")), None
        )
        if base_anchor is not None:
            ax = float(base_anchor.get("x") or bx)
            ay = float(base_anchor.get("y") or by)
        else:
            ax, ay = bx, by
        # The build site is chosen **once** and then reused for the whole episode.
        #
        # Recomputing "the grid cell nearest the builder" every decision looks
        # harmless and is not: the policy's own tactical orders keep moving the
        # builder, so the nearest cell (and therefore the ordered site) drifts with
        # it and the factory restarts at a new location every decision, forever.
        # Measured on a masked-random Lake episode (``.runtime/interference.json``,
        # curriculum stage already on ``build_structure``): the builder wandered
        # x 1091 -> 1082, y 2055 -> 2119 over 40 decisions while every decision
        # re-ordered the factory at the new nearest cell -- 2006 decisions later
        # the factory was still at ``build_progress < 1`` and ``bonus_produce`` was
        # 0.0. Ordering the *same* site repeatedly does finish (single vs repeated
        # order: tick 1891 vs 2611), so latching the site is the whole fix.
        if self.factory_site is None:
            grid = int(self.micro.position_grid)
            # Only *other factories* block a candidate, not every building.
            #
            # With every own structure counted, all nine candidates were filtered out
            # on Lake -- the command centre sits at the anchor and the clearance radius
            # (200 px) excluded the whole ring around it -- so the code fell through to
            # the grid scan and dispatched ``grid_to_world(3, 5)`` =
            # ``(1117.71, 1775.43)``, which is **water**. The engine accepted the order
            # and silently dropped it, so no factory ever existed. Measured:
            # ``building_types_ever_seen`` was only command centre + extractors, and
            # every episode fielded a single ``scout``.
            #
            # What must be avoided is two factories in one place (the engine would
            # refuse the second). Overlapping an extractor or the main building is
            # handled by the engine's own placement rule, which the bridge now enforces
            # as an honest rejection rather than a false acceptance.
            occupied: list[tuple[float, float]] = [
                (float(unit.get("x") or 0.0), float(unit.get("y") or 0.0))
                for unit in own
                if str(unit.get("unit_type") or "").lower() == "landfactory"
            ]
            # Prefer an exact, engine-verified point over a quantised grid cell.
            #
            # The grid has 8 cells over 2600 px, i.e. ~364 px per step, and the cells
            # nearest our base land **in the lake**: measured
            # (``.runtime/build_receipts.json``) every ``build_structure`` went to
            # ``(1117.71, 1775.43)``, which the engine silently drops -- the receipt
            # still said ``accepted``. The result was that no ``landFactory`` ever
            # existed in 6 of 6 episodes and the whole army was a single ``scout``.
            #
            # The offsets below are the ones verified against the engine in
            # ``.runtime/turret_sites.json`` (a turret built at each of them), i.e.
            # known-legal land inside our own quarter. They are tried first, nearest
            # the builder first; the grid scan stays as the fallback for maps where
            # none of them is legal, so this cannot break a different map.
            exact_candidates = [
                (ax + dx, ay + dy)
                for dx, dy in (
                    (0.0, 0.0), (90.0, 0.0), (-90.0, 0.0),
                    (0.0, 120.0), (0.0, -120.0), (140.0, 120.0),
                    (-140.0, 120.0), (140.0, -120.0), (-140.0, -120.0),
                )
            ]
            chosen: tuple[int, int] | None = None
            # Skip candidates already proven illegal on this map.
            #
            # A site the engine drops without creating a hull can only be identified
            # by trying it: the bridge's placement gate passes, the receipt says
            # ``accepted``, and the hull never appears (``.runtime/factory_site_trace.json``:
            # ``[1090.897, 2069.848]`` ordered 24 times, 0 hulls; the bridge's own debug
            # trace says ``ordered=1 at (1090,2069)``, so nothing rejects it at the
            # bridge). The list is what makes the rotation below converge instead of
            # cycling back onto a dead cell.
            dead = getattr(self, "_factory_dead_sites", set())
            usable = [
                (wx, wy) for wx, wy in exact_candidates
                if (round(wx, 1), round(wy, 1)) not in dead
            ]
            # Start the ladder at the first untried candidate rather than at "nearest
            # the base" every time, so a confirmed-dead site costs one block of stall
            # decisions rather than the whole episode.
            offset = int(getattr(self, "_factory_candidate_offset", 0)) % max(1, len(usable))
            usable = usable[offset:] + usable[:offset]
            for wx, wy in usable:
                if any(
                    (wx - ox) ** 2 + (wy - oy) ** 2 < 200.0 * 200.0
                    for ox, oy in occupied
                ):
                    continue
                # Keep it on the map.
                if not (0.0 <= wx <= float(self.micro.map_width)):
                    continue
                if not (0.0 <= wy <= float(self.micro.map_height)):
                    continue
                cgx = int(round(wx / max(1.0, float(self.micro.map_width)) * (grid - 1)))
                cgy = int(round(wy / max(1.0, float(self.micro.map_height)) * (grid - 1)))
                chosen = (max(0, min(grid - 1, cgx)), max(0, min(grid - 1, cgy)))
                self.factory_exact = (wx, wy)
                break
            if chosen is None:
                best = (0, 0)
                best_d = float("inf")
                for gy in range(grid):
                    for gx in range(grid):
                        wx, wy = self.micro.grid_to_world(gx, gy)
                        if any(
                            (wx - ox) ** 2 + (wy - oy) ** 2 < 260.0 * 260.0
                            for ox, oy in occupied
                        ):
                            continue
                        d = (wx - bx) ** 2 + (wy - by) ** 2
                        if d < best_d:
                            best_d = d
                            best = (gx, gy)
                chosen = (int(best[0]), int(best[1]))
                self.factory_exact = self.micro.grid_to_world(*chosen)
            self.factory_site = chosen
        gx, gy = self.factory_site
        # Re-issue only after a stall, at the **latched** site.
        #
        # The latched site is required: recomputing "the nearest cell" each decision
        # made the target follow the builder around the map (x 1091 -> 1082,
        # y 2055 -> 2119 across 40 decisions), so the factory restarted somewhere
        # new every decision and never completed.
        #
        # The stall gate is required for the reason measured in
        # ``.runtime/ab_reissue.py``: repeating a build order for the same target
        # restarts it and pins ``build_progress`` at 0.0 forever. The earlier
        # "re-issue every decision" rule made sense only if a repeat were a no-op,
        # which it is not. See :data:`STALL_DECISIONS`.
        progress_now = None
        if unfinished:
            try:
                progress_now = float(unfinished[0].get("build_progress") or 0.0)
            except (TypeError, ValueError):
                progress_now = None
        if progress_now is not None:
            previous = getattr(self, "_factory_progress", None)
            if previous is None or progress_now > float(previous) + 1e-9:
                self._factory_progress = progress_now
                self._factory_stalled = 0
                return []
            self._factory_stalled = int(getattr(self, "_factory_stalled", 0)) + 1
            if self._factory_stalled < STALL_DECISIONS:
                return []
            self._factory_stalled = 0
        else:
            # No hull exists at all. That is NOT "everything is fine" -- it is the
            # signature of a site the engine silently drops, and it used to be
            # indistinguishable from "the order is about to be accepted".
            #
            # The stall gate above only runs when an unfinished hull is visible, so a
            # site that never produces one never got promoted to "stalled". Measured
            # on Lake at difficulty 1 with `economy_max_factories = 5`: the stage sent
            # ``[1090.897, 2069.848]`` **24 times**, every receipt said ``accepted``,
            # no hull ever appeared, and the factory count stayed at exactly 1 for the
            # remaining 18600 ticks (``.runtime/factory_site_trace.json``). The bridge's
            # own debug trace agrees the order was dispatched
            # (``ordered=1 at (1090,2069)``), so the refusal happens inside the engine
            # after the bridge's placement gate -- it can only be detected by outcome,
            # and only by counting orders that produced nothing.
            #
            # Consecutive orders at a latched site with no hull means the site is dead:
            # retire it and let the ladder advance.
            orders_here = int(getattr(self, "_factory_orders_at_site", 0)) + 1
            self._factory_orders_at_site = orders_here
            if orders_here >= STALL_DECISIONS:
                dead = getattr(self, "_factory_dead_sites", None)
                if dead is None:
                    dead = set()
                    self._factory_dead_sites = dead
                if self.factory_exact is not None:
                    dead.add((round(self.factory_exact[0], 1),
                              round(self.factory_exact[1], 1)))
                self._factory_candidate_offset = int(
                    getattr(self, "_factory_candidate_offset", 0)
                ) + 1
                self.factory_site = None
                self.factory_exact = None
                self._factory_orders_at_site = 0
                self._factory_progress = None
                self._factory_stalled = 0
                return []
        # A hull appeared (or progressed): this site works, so forget the counter.
        self._factory_orders_at_site = 0
        action = np.array(
            [ACTION_KINDS.index("build_structure"), int(self.micro.groups),
             int(gx), int(gy), 0],
            dtype=np.int64,
        )
        # Encode the structure type the curriculum actually wants.
        #
        # The target head used to mean nothing for ``build_structure`` (the type
        # was hard-coded to ``land_factory``), so this slot was a free 0. It is now
        # the *structure menu index*, which means leaving it at 0 would ask for
        # ``build_menu(state)[0]`` -- and on a map whose builder offers
        # ``["landFactory", "airFactory", ...]`` that happens to be right, while on
        # one that offers ``["turret", ...]`` the curriculum would silently build a
        # turret instead of the factory its whole downstream depends on.
        #
        # Resolving the index from the menu is what keeps the *order* the same as
        # before on today's maps while making it explicit rather than accidental.
        menu = build_menu(self.obs_state)
        if menu:
            action[4] = int(menu_index(menu, "land_factory"))        # Bound the number of factory *orders* per episode.
        #
        # This must sit here, after every early return, and be counted only when an
        # order is actually about to be sent.
        #
        # The reason is a real bug that this placement fixes: when the counter was
        # incremented near the top of the method, every decision that merely *waited*
        # (for the opening extractor, for a structure to finish) burned one unit of the
        # budget. Measured: ``build_structure`` was issued exactly **24 times** -- the
        # whole allowance -- and then the stage stopped for the rest of the episode, so
        # **no factory was ever built** and every episode fielded only ``scout``
        # (``.runtime/trained_behaviour.json``). The count must measure orders, not
        # calls.
        #
        # The hazard it guards is genuine: when the chosen site is rejected by the
        # engine, no hull ever appears, so nothing can be observed to stall and the
        # stage would re-order on every decision (measured **1736** orders and 4/4
        # timeouts, ``.runtime/dispatch_tap.json``).
        # The budget scales with how many factories *already exist*, not with the
        # quota alone.
        #
        # A fixed allowance of ``max_factories * 8 + 8`` is spent on failed
        # placements, and failed placements are the normal state early on: the engine
        # rejects a site (terrain, clearance, an unfinished neighbour) and no hull
        # appears, so nothing can be observed to stall and the stage must retry. Only
        # when a hull exists does the site latch and the retries stop.
        #
        # Measured with ``max_factories = 5`` on Lake at difficulty 1: the whole
        # 48-order allowance was gone by **tick 1411**, before the FIRST factory was
        # finished (it completed at tick 2011), and the count then stayed at exactly 1
        # for the remaining 18600 ticks -- the stage was not refusing to build the
        # second, third or fourth factory, it had no allowance left to try
        # (``.runtime/factory_blockers.json``, where ``OPEN(orders_used=48)`` appears
        # with ``max factories ever: 1``).
        #
        # So the allowance is per *achieved* factory: one factory's worth of retries
        # is enough to place one factory, and every completed factory unlocks the next
        # block. A stage that cannot place anything at all still stops after one
        # block, which is the protection the original counter was added for (measured
        # **1736** re-orders and 4/4 timeouts without any bound,
        # ``.runtime/dispatch_tap.json``).
        built_factories = len(operational)
        max_orders = (built_factories + 1) * (int(self.macro.max_factories) * 8 + 8)
        if int(self._factory_orders) >= max_orders:
            return []
        self._factory_orders = int(self._factory_orders) + 1
        # Emit the **exact** world coordinate, not the 8x8 grid cell.
        #
        # ``decode_action`` quantises through ``grid_to_world``, and that quantisation
        # is what put every factory order into the lake: measured
        # (``.runtime/build_receipts.json``) the dispatched coordinate was
        # ``(1117.71, 1775.43)`` while the chosen candidate was land, so the engine
        # dropped the order and no ``landFactory`` ever existed.
        #
        # The grid exists for the *policy*, whose position head has 8 cells. The
        # curriculum knows its point exactly and must not lose it -- the same fix that
        # ``_issue_defence`` needed for the same reason.
        wx, wy = self.factory_exact or self.micro.grid_to_world(int(gx), int(gy))
        structure_type = menu[action[4]] if menu else "land_factory"
        self.curriculum_order = {
            "kind": "build_structure",
            "structure_type": structure_type,
            "grid": [int(gx), int(gy)],
            "world": [round(float(wx), 2), round(float(wy), 2)],
            "accepted": True,
        }
        return [
            {
                "kind": "build_structure",
                "unit_group": int(self.micro.groups),
                "structure_type": structure_type,
                "x": round(float(wx), 2),
                "y": round(float(wy), 2),
            }
        ]

    def _curriculum_actions(self) -> list[dict[str, Any]]:
        """Orders the opening curriculum adds to the *current* dispatch.

        Why they ride along instead of being sent as a second message
        ------------------------------------------------------------
        A second ``step`` message is a second dispatch on a later tick, not an
        extra order in the same decision. Measured on a masked-random Lake episode
        (``.runtime/extractor_trace.json``, 220 decisions): the curriculum
        re-issued ``build_extractor`` on every decision and the extractor's
        ``build_progress`` stayed at exactly **0.0000** for all 220 -- the policy's
        own tactical order (``attack_move`` / ``move_group`` / ``stop``, about half
        of all sampled kinds) reached the builder first, cancelled the in-flight
        construction, and only then did the curriculum re-issue it. Progress was
        reset before it could accumulate, so the build never completed and nothing
        downstream could fire. The same effect is produced by two *curriculum*
        stages targeting the same builder: they would cancel each other.

        That is why the stages are **strictly ordered and mutually exclusive** --
        at most one of them may spend the builder in a given decision, and the
        earlier stage wins. Building the factory before expanding the economy is
        the correct priority: an army is what defends the economy, and a map full
        of undefended extractors just feeds the enemy.

        Measured cost of getting this wrong (``.runtime/production_trace.json``):
        with expansion allowed to grab the builder, the factory stage never ran --
        ``no_combat_factory`` was the blocking reason on **1968 of 1968** decisions,
        the episode ended with 5 extractors and 19748 unspent credits, and the army
        was still empty.

        The three stages:

        1. ``build_extractor`` while the opening extractor is not finished;
        2. ``build_structure`` (land factory) once it is;
        3. ``produce_unit`` once a factory advertises a non-builder menu entry;
        4. ``build_extractor`` again (expansion), only when 2 and 3 have nothing
           to do -- a finished factory exists and its queue is empty.

        Every order goes through the normal bridge path, and the rewards
        (``bonus_build`` / ``income`` / ``bonus_produce``) are still paid only by
        the observation stream when a structure actually finishes or a new combat
        unit actually appears -- so this removes an exploration barrier without
        inventing reward, and repeating an order cannot farm anything.
        """

        if getattr(self, "_curriculum_disabled", False):
            return []
        # ``curriculum_mode`` decides whether the stages may *act* at all.
        #
        # This is the switch that turns the economy from a script into a decision.
        # Under ``"orders"`` (the historical default) the stages below issue real
        # orders, so the "agent" that scores well may be scoring on the strength of
        # the script -- and measured, that arrangement hurt more than it helped: the
        # policy's own orders cancelled the stage's in-flight construction and the
        # trained policy reached 43.75% against **93.75% for a policy that issued
        # nothing** (``.runtime/ppo_interference.json``).
        #
        # ``"mask"`` keeps every bit of the stages' knowledge about what is *legal*
        # (it lives in the action masks, which are derived from the engine's own
        # reported capabilities) but emits no order, no matter what the stages
        # think should happen next. ``"off"`` disables both. Returning ``[]`` here
        # rather than skipping the call sites means there is exactly one place
        # where "the script may act" is decided, so a new stage cannot accidentally
        # opt back in.
        if not getattr(self.game, "curriculum_issues_orders", True):
            self.curriculum_order = None
            return []
        # Cleared each decision so ``curriculum_order`` always describes *this*
        # step. Leaving it set made the field report the last stage that ever fired,
        # which is how a finished extractor and an un-ordered factory looked
        # identical in a trace and sent the diagnosis down the wrong path.
        self.curriculum_order = None
        # Multiple stages may contribute to one dispatch, but only in the specific
        # pairing described at the ``_order_blocks_later_stages`` check below.
        collected: list[dict[str, Any]] = []
        # Order matters and is a **priority**. Two different rules apply, and both
        # are measured rather than stylistic:
        #
        # 1. **Upgrades come first.** A T1 -> T2 extractor is a *permanent* increase
        #    in the rate that pays for every future unit; one more tank is not. It
        #    also has to be evaluated before any ``build_*`` stage, or it is starved:
        #    a build stage returns a blocking order and the loop stops, so a stage
        #    placed after it is never reached. Measured
        #    (``.runtime/stage_priority.json``, 447 decisions): with the upgrade stage
        #    placed after the factory stage its gates opened on 6 decisions and it was
        #    reached on none of them. It does not compete for the builder -- it
        #    addresses a building -- so running it first costs nothing and
        #    ``_order_blocks_later_stages`` lets the builder stages follow in the same
        #    dispatch.
        # 2. **Expansion sits before the attack.** With the attack stage ahead of it,
        #    the army is large enough to march for most of the episode, so the attack
        #    order was produced first on nearly every decision and expansion never
        #    ran: occupied pools stayed at **1** and the win rate was 8-16%. Put in
        #    front, expansion claims the builder whenever the factory chain is idle,
        #    and the attack order rides along in the same dispatch.
        #
        # That order is now the *default* rather than the only possibility: a
        # commander's ``MacroPlan.build_order`` decides which stages run and in what
        # priority (see :mod:`ironzero.rwpp.macro`). The default is preserved
        # exactly, so a run with no directive -- or an empty plan -- walks this list
        # in the same order as before.
        #
        # The mapping is local on purpose: it is read on every decision, and a
        # module-level table would have to be defined after the class body (the
        # methods do not exist until then) which is exactly the kind of ordering
        # hazard that breaks on the day somebody moves a method.
        stages = {
            "upgrade": self._issue_upgrades,
            "extractor": self._issue_opening_extractor,
            "factory": self._issue_opening_factory,
            "production": self._issue_opening_production,
            "defence": self._issue_defence,
            "expansion": self._issue_late_expansion,
            "attack": self._issue_attack_drive,
        }
        for name in self.macro.stage_names:
            stage = stages.get(name)
            if stage is None:  # pragma: no cover - MacroPlan validates names
                continue
            order = stage()
            # ``order`` is a *list* of decoded orders. Testing it for ``None``
            # would be true for an empty list too -- and that is exactly the bug
            # this line used to have: the extractor stage returns ``[]`` once it is
            # satisfied, so the loop short-circuited on it forever and the factory
            # and production stages were never reached. Measured on a pure ``noop``
            # Lake episode with the extractor finished at tick 2001: the extractor
            # stage was called 119/119 times and returned ``[]`` every time, the
            # factory stage was never called, and ``bonus_produce`` stayed 0.0.
            if order:
                collected.extend(order)
                # The **build** stages own the single builder and must stay
                # mutually exclusive: two build orders in one dispatch would cancel
                # each other, which is the same failure as re-issuing one order every
                # decision (measured: ``build_progress`` pinned at 0.0 forever).
                #
                # The **attack** stage is different -- it addresses a combat group
                # that excludes the builder -- so it may ride along with a build
                # order. Without that, an army large enough to march makes the attack
                # stage fire every decision, which (before this change) starved the
                # economy: occupied pools stayed at **1** and the win rate collapsed
                # from 66.7% to 8-16%.
                #
                # The **upgrade** stage is different again, and for a third reason:
                # it addresses a *building*, not the builder. An extractor tier-up
                # does not compete for the builder's attention, so an upgrade and a
                # construction can be in flight at the same time.
                #
                # That distinction is load-bearing and was measured
                # (``.runtime/stage_priority.json``, 447 live decisions): with
                # upgrades treated as an ordinary builder order, the stage's own
                # gates opened on 6 decisions but it **won the dispatch 0 times**,
                # because ``_issue_late_expansion`` or ``_issue_defence`` fired first
                # on every other decision and the loop returned before reaching it.
                # From the outside that is indistinguishable from "the stage is
                # broken" -- which is exactly how "never upgrades" looked.
                #
                # The guard that keeps this safe is inside the stage itself: it
                # refuses to fire while any of our structures is unfinished, so it can
                # never overlap with a construction order in practice.
                if self._order_blocks_later_stages(order):
                    # Return **everything collected so far**, not just this stage's
                    # order. ``collected`` may already hold a non-blocking order from
                    # an earlier stage (an upgrade, or the attack ``attack_move``),
                    # and returning ``order`` alone silently discarded it.
                    #
                    # That was a live bug, not a hypothetical one: measured in
                    # ``.runtime/stage_priority.json``, ``_issue_upgrades`` produced a
                    # legal ``upgrade_building`` on 6 decisions, and on every one of
                    # them ``_issue_opening_production`` fired afterwards and returned
                    # only its own ``produce_unit`` -- so the upgrade never reached the
                    # bridge and "never upgrades" stayed true even after the stage and
                    # its gates were fixed.
                    return collected
        return collected

    @staticmethod
    def _order_blocks_later_stages(order: list[dict[str, Any]]) -> bool:
        """Whether this order must not share a dispatch with later stages.

        ``True`` means "stop here" -- the order competes for the one builder (or the
        one factory queue), so a second stage adding its own order in the same
        dispatch would make the two cancel each other. That is the failure measured in
        ``.runtime/ab_reissue.py`` (``build_progress`` pinned at 0.0 forever while
        every receipt said ``accepted: true``), and it is why this loop returns early
        at all.

        ``False`` means "a later stage may still contribute", which is only true for
        orders that address *something other than the builder*:

        * tactical orders (``move`` / ``attack_*``) address combat groups;
        * ``upgrade_building`` addresses an existing building.

        Getting this predicate inverted is not a cosmetic bug. Measured
        (``.runtime/dispatch_tap.json``) with the polarity reversed, a single dispatch
        carried **three** construction orders::

            build_structure:turret  +  build_structure:landFactory  +  build_extractor

        all aimed at the single builder. Each cancels the previous one, so none of them
        ever accumulated progress and ``turret_count_ever`` stayed 0 while the tap
        proved the turret order *was* being sent -- the last cancelling order won.
        """

        # ``produce_unit`` is **not** shared with the builder goals.
        #
        # It used to be, and that made production and the march mutually exclusive
        # for a reason that does not hold: ``produce_unit`` addresses a *factory
        # queue*, not the single builder. A factory can be filling its queue while
        # the builder constructs an extractor and the army marches -- three
        # different actors.
        #
        # The measured cost of the old grouping (``.runtime/production_plateau.json``
        # on Lake at difficulty 1): the production stage reported ``ORDERED`` on
        # **1157 of 1400** decisions, and because every one of those blocked later
        # stages, the attack stage never ran. The army therefore never grew past the
        # ``attack_at_army`` gate it was waiting on, and the episode timed out with
        # the army idle at home -- the exact passivity the attack stage exists to
        # prevent.
        #
        # The two *build* goals stay together: they do compete for the one builder,
        # and two construction orders in one dispatch cancel each other (measured:
        # ``build_progress`` pinned at 0.0 forever while every receipt said
        # ``accepted: true``).
        shared = {"build_extractor", "build_structure"}
        return any(
            str(entry.get("kind") or "") in shared
            for entry in order
            if isinstance(entry, dict)
        )

    @staticmethod
    def _order_is_tactical(order: list[dict[str, Any]]) -> bool:
        """Whether a curriculum order is a *movement* order rather than a build.

        Retained because ``_order_blocks_later_stages`` generalises it and callers
        may still want the narrower question. Movement orders are safe to combine
        with a build order in the same dispatch, because the attack stage addresses a
        combat group that excludes the builder (see :meth:`_issue_attack_drive`).
        Build orders are not: two of them in one dispatch cancel each other, and
        repeating one at all resets the construction (``.runtime/ab_reissue.py``).
        """

        return any(
            str(entry.get("kind") or "") in {"move_group", "attack_move", "attack_target"}
            for entry in order
            if isinstance(entry, dict)
        )

    def action_masks(self) -> dict[str, np.ndarray]:
        return self.masks

    def close(self) -> None:
        self._shutdown_game()

    def __enter__(self) -> "RwppDuelEnv":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Process / bridge plumbing
    # ------------------------------------------------------------------
    def _start_game(self) -> None:
        start_game(self)

    def _await_observation(self, timeout: float) -> dict[str, Any] | None:
        return await_observation(self, timeout)

    def _drain_verdict(self) -> dict[str, Any] | None:
        return drain_verdict(self)

    def _shutdown_game(self) -> None:
        shutdown_game(self)


__all__ = ["RwppDuelEnv"]





