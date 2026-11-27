# SPDX-License-Identifier: Apache-2.0
"""Configuration for the deterministic micro-duel reference environment.

The mock engine exists so the whole IronZero loop (bridge protocol ->
observation encoding -> action masks -> PPO -> evaluation -> replay) can be
built and tested without the real game client. Every constant here is a
contract shared by the engine, the encoders and the training code.
"""

from __future__ import annotations

from dataclasses import dataclass


def _grid_index_to_world(index: int, grid: int, size: float, margin: float) -> float:
    if grid <= 1:
        return size / 2.0
    span = size - 2.0 * margin
    return margin + span * index / (grid - 1)


@dataclass(frozen=True, slots=True)
class MicroDuelConfig:
    """Fixed micro-duel scenario: 8 tanks per side, no economy, no fog.

    Defaults are calibrated from Rusted Warfare 1.15 ``assets/units/tanks/tank.ini``
    (HP 210, direct damage 25, range 130, shoot delay 75 ticks, move speed 1.1
    units/tick at a 60 Hz simulation loop) and a 1600x1600 fixed micro arena.
    """

    # Simulation clock (Rusted Warfare's fixed loop runs at 60 ticks/s)
    tick_rate: int = 60
    decision_interval: int = 30  # one model decision every 0.5 s
    episode_ticks: int = 3600  # 60 s episode

    # Map (small symmetric training arena; the real Lake map is 2600x2600)
    map_width: float = 1000.0
    map_height: float = 1000.0
    spawn_margin: float = 220.0

    # Units (real 1.15 tank values)
    teams: int = 2
    units_per_team: int = 8
    groups: int = 4
    unit_hp: float = 210.0
    unit_damage: float = 25.0
    attack_range: float = 130.0
    attack_cooldown_ticks: int = 75
    unit_speed: float = 66.0  # world units per second (1.1 units/tick * 60)
    unit_radius: float = 11.0
    unit_price: float = 350.0

    # Fog of war: 0 disables it (default). When > 0, enemies farther than this
    # distance from every friendly unit are hidden from observations/targeting.
    vision_radius: float = 0.0
    # Keep the builder out of the policy's tactical group head.
    #
    # Every real order cancels the in-flight one, so a tactical order addressed to
    # the builder's group stops whatever it was constructing. Measured
    # (``.runtime/ppo_interference.json``): a policy issuing one order per decision
    # sends it while a structure is unfinished on **1949 of 1949** decisions, the
    # episode ends with ``occupied = 1`` / ``income = 26`` / 3 entities, and its win
    # rate was 43.75% -- against 93.75% for a policy that issues **nothing** with the
    # same curriculum. The policy was destroying the economy that builds its army.
    #
    # ``True`` closes groups that contain *only* builders (and the broadcast slot
    # when every living unit is a builder). Mixed groups stay open, because closing
    # them would disable part of the army. Set ``False`` to reproduce the
    # unshielded behaviour and measure the difference.
    protect_builder: bool = True
    # Let the curriculum own the two *build* action kinds while it is running.
    #
    # Closing the builder's tactical group is not sufficient: ``build_extractor``
    # and ``build_structure`` decode into orders that address the builder directly,
    # so the policy can still cancel construction through the kind head. Measured
    # with only the group protected: ``occupied = 1``, ``income = 26``, three
    # entities at the tick cap (``.runtime/mask_reach.json``).
    #
    # The curriculum issues exactly those two kinds, so re-learning them adds no
    # decision the environment is not already making, while every duplicate order
    # resets the build. With this on, the agent's decisions are the ones the
    # curriculum does not make: what to produce, where to go, whom to attack.
    #
    # Off by default: it is only meaningful together with an active curriculum, and
    # turning it on changes what the action space *means*, so it must be opt-in
    # rather than silently true. ``RwppGameConfig.micro_config`` enables it when the
    # opening curriculum is on.
    curriculum_owns_build_kinds: bool = False
    # How many independent action *slots* one decision carries.
    #
    # ``1`` -- the historical layout: one flat vector, so one decision issues at
    # most one order. That makes the economy and the army mutually exclusive at the
    # decision level: a ``build_extractor`` and a ``move_group`` can never be
    # chosen together, and neither can ``produce_unit`` and ``attack_move``. Every
    # order also addresses the builder or the army, and *every* real order cancels
    # the in-flight one, so a tactical decision taken while the builder is working
    # silently destroys construction. Measured end to end: the trained policy
    # reached 43.75% while a policy that issued **nothing** reached 93.75% on the
    # same curriculum -- the policy's measured contribution was negative, because
    # it kept cancelling the economy that builds its own army
    # (``.runtime/ppo_interference.json``, and the note in ``rwpp/stepping.py``).
    #
    # ``2`` -- the two-slot layout: an **economy** slot and a **tactical** slot,
    # each with its own kind vocabulary (``rwpp.decoding.ECONOMY_ACTION_KINDS`` /
    # ``TACTICAL_ACTION_KINDS``) and each able to say "nothing this decision". One
    # decision therefore expresses "keep building the factory AND move group 2"
    # -- which is what a human player does, and what makes the economy a *decision*
    # rather than something a scripted stage performs on the agent's behalf.
    #
    # Default ``1`` on purpose: this changes what the action vector *means*, so a
    # checkpoint saved under one layout cannot be interpreted under the other. An
    # opt-in flag keeps every existing checkpoint, test and measurement valid, and
    # makes the A/B honest (same policy class, same reward, one variable).
    action_slots: int = 1

    # Observation / action shape
    #
    # ``max_entities`` is the **enemy visibility budget**. The default stays ``8``
    # because it is the *historical* layout: every shipped checkpoint, the mock
    # engine's fixtures and the reference measurements are expressed in it, and
    # changing it silently would reinterpret all of them
    # (``tests/test_two_slot_actions.py::test_default_is_the_old_layout`` pins this
    # contract deliberately).
    #
    # ``8`` is also too small to play on, and that is a measured defect rather than
    # a preference. On ``[p2]Lake (2p)`` a 1200-frame replay
    # (``runs/platform/real-rwpp-20261116-133006-d97c5a``) dropped **26.1%** of all
    # enemy unit-frames, and **61.7% of the slots that were kept went to buildings**
    # -- because the enemy block is deliberately *not* reordered (the target head
    # indexes it, see ``observation.encoding``), so it keeps the engine's spawn
    # order, in which the map-placed command centre, builders, factories and
    # extractors all have lower ids than every produced combat unit.
    #
    # The consequence was not subtle: ``c_artillery`` -- 290 px range against a
    # tank's 130, i.e. the unit that decides an exchange -- was visible on **0 of
    # 500** unit-frames it existed for. "The policy does not focus the enemy's high
    # output units" measured as an *observation* defect, not a strategy one.
    #
    # The fix therefore lives in the **training/eval entry points**
    # (``training.train_rwpp``, ``training.rwpp_ppo``, ``scripts.eval_rwpp_policy``),
    # which default to :data:`WIDE_MAX_ENTITIES` and stay consistent with each other
    # so a checkpoint's layout is always reproduced at inference time. Raising it
    # here instead would have broken every existing checkpoint for no gain, since a
    # run that wants the narrow layout can still ask for it explicitly.
    max_entities: int = 8
    # ``target_slots`` is the shared target head: enemy slots + menu entries +
    # the "no target" fallback. It must be at least ``max_entities + 1`` or the
    # policy can see an enemy it cannot name; the encoder and the mask both size
    # themselves from this number.
    target_slots: int = 9  # 8 enemy slots + "no target"
    position_grid: int = 8
    map_channels: int = 3
    map_grid: int = 8
    # Global feature block. The last three entries carry the economy state the
    # policy needs to decide *whether* it can afford a build action:
    #   [funds / funds_scale, income / income_scale, extractor_slots_owned]
    # They are appended to the original 12 so the layout stays a superset.
    global_dim: int = 15
    entity_features: int = 10
    # Per-entity *identity* features, written as a **separate block after the
    # control grid** rather than inside ``entity_features``.
    #
    # The ten ``entity_features`` are geometric (position / hp fraction / order /
    # distance) and contain no type information at all, so two completely
    # different situations encode to a bit-identical vector. Measured (see
    # ``runs/_scratch/probe_obs_type_blind.py``): "commandCenter 4000HP + builder
    # 170HP" and "tank 4000HP + scout 170HP" both produce the *same* 367-float
    # observation, max per-element difference 0.0. The policy therefore could not
    # tell a building from a vehicle, a factory from an extractor, or a builder
    # from a tank -- which is why "build something" was never learnable.
    #
    # Layout choice matters for checkpoints: appending the block at the tail
    # means the global / own / enemy / grid columns keep their exact previous
    # offsets, so a checkpoint saved before this block existed keeps its meaning
    # and only the new tail starts from zero influence. Inserting the new
    # features inside each entity's stride would have shifted every later column
    # and silently scrambled an old checkpoint.
    #
    # NOTE: that argument covers *this* block only. ``global_dim`` grew 12 -> 15
    # earlier and it sits at the *head*, so every column after it moved by three.
    # Migration therefore cannot be a plain "copy the leading columns" prefix
    # (see ``training.ppo._column_source``): columns are re-mapped block by block
    # so each weight stays attached to the feature it was trained on.
    #
    # 7 -> 9: two attack capabilities appended at the *tail* of each entity's
    # stride, so the first seven columns of every slot keep their exact meaning
    # and an older checkpoint's weights stay attached to the features they were
    # trained on. Before this the policy could describe where an entity was and
    # how much hp it had, but not whether it could shoot -- observed as a builder
    # walking straight into the enemy base to die. Columns 7/8 are
    # ``can_attack`` and ``attack_range / ATTACK_RANGE_SCALE``.
    #
    # 9 -> 17: eight more columns, again appended at the tail, for the attributes
    # a real-time-strategy decision actually needs and the bridge did not expose:
    #
    # ==== ==============================================================
    # 9    ``move_speed / MOVE_SPEED_SCALE``  -- 追不追得上 / 跑不跑得掉
    # 10   ``tech_level / 3``                 -- 等级: 对方是不是已经升了 T2/T3
    # 11   ``can_attack_air``                 -- 能不能打飞机 (空军是否安全)
    # 12   ``is_defence``                     -- 防御建筑: 静止的威胁
    # 13   ``shield / max_shield``            -- 护盾, 先要打掉的那层
    # 14   ``shoot_delay / SHOOT_DELAY_SCALE``-- 射速 (真实 DPS 的一半信息)
    # 15   ``queue_length / QUEUE_SCALE``     -- 建造队列: 工厂在造几个
    # 16   ``queue_progress``                 -- 建造队列: 当前这个造到哪了
    # ==== ==============================================================
    #
    # Column 9 answers "can I catch it / can I outrun it", which no other column
    # can: a turret and a tank with identical hp and range are opposite problems.
    # Columns 15/16 are the bridge's answer to "what is this factory producing",
    # the information whose absence made behaviour cloning impossible -- the
    # built-in AI's army grows from 2 to 13 units while the recorded order stream
    # contains zero production orders, because production lives in the build queue
    # rather than in waypoints.
    #
    # Note on 防御/armour: this engine has no per-unit armour *stat*. Defence is
    # (a) a shield pool (columns 13) and (b) per-*projectile* damage multipliers
    # (``hullDamageMultiplier`` etc. on ``Projectile``). There is therefore no
    # honest "armour number" to emit per unit, and inventing one would be worse
    # than omitting it. Shield plus ``can_attack_air`` plus range is the real
    # defensive profile the engine exposes.
    type_features: int = 17
    # Fixed module-level denominator for the ``attack_range`` column (mirrored in
    # ``observation.features.ATTACK_RANGE_SCALE``). It is deliberately *not* a
    # config field: the map width changes per scenario, and a scenario-dependent
    # denominator would make the same tank read differently on Lake than on the
    # micro arena. 1000 px spans the whole stock 1.15 range ladder (tank 130,
    # artillery 290, turrets < 400); longer ranges clamp to 1.0 rather than break
    # the block's [0, 1] contract.
    # Normalisation denominators for the economy features. Fixed constants (not
    # derived from the scenario) so a checkpoint stays valid across maps.
    funds_scale: float = 5000.0
    income_scale: float = 40.0
    # Optional terrain passability layer. When set, ``map_channels`` should be 4
    # (three army channels + one static terrain channel) and the encoder reads
    # the blocked-tile grid from this JSON file (TerrainGrid ``map.json`` format).
    # ``None`` keeps the legacy 3-channel observation and the old ``obs_dim``.
    terrain_path: str | None = None

    # Reward shaping
    potential_coef: float = 0.5
    win_reward: float = 1.0
    loss_reward: float = -1.0

    # Economy and Macro extension
    enable_economy: bool = True
    starting_funds: float = 1000.0
    base_income_per_second: float = 1.0
    extractor_income_per_second: float = 8.0
    extractor_t2_income_per_second: float = 18.0
    extractor_cost: float = 700.0
    #: Price the *engine* charges to tier an extractor up.
    #:
    #: 1400, not 1200. This is not a tuning knob: it is the engine's own
    #: ``SpecialAction.getPriceValue()`` for the extractor's
    #: ``getBuildActionId()`` action, read out of the live game by the bridge.
    #: Measured on ``[p2]Lake (2p)`` at difficulty 1 (round 19):
    #:
    #:   [ironzero] upgrade rejected: building=25 available=true
    #:             affordable=false price=1400
    #:
    #: The old 1200 made every gate that used this field under-reserve by 200, so
    #: the upgrade stage ordered an upgrade the engine then refused as
    #: unaffordable, and no building ever left tech_level 1.
    extractor_upgrade_cost: float = 1400.0
    factory_cost: float = 800.0
    turret_cost: float = 500.0
    #: Nominal price of "one combat unit" for the ``produce_unit`` mask gate.
    #:
    #: **Not** the cheapest real unit: on a stock land factory the cheapest is
    #: ``tank`` at 350 (``hoverTank`` 450, ``artillery`` 900). See
    #: :data:`UNIT_PRICES` for the real table, which the mask now prefers when
    #: the bridge reports a menu. This value remains the fallback for
    #: environments that report no menu at all (mock engine, older bridge).
    tank_cost: float = 250.0
    resource_slots_count: int = 4

    # Economic reward & penalty weights
    extractor_reward: float = 0.5
    extractor_t2_reward: float = 0.8
    factory_reward: float = 0.6
    production_reward: float = 0.4
    building_loss_penalty: float = 1.0
    idle_funds_penalty_coef: float = 0.0001
    # Funds above this threshold are considered idle capital and are penalised.
    # Derived from the scenario's own price list rather than a magic number so
    # the penalty actually becomes reachable in short episodes.
    idle_funds_threshold: float = 800.0

    # Episode control
    opponent: str = "scripted"  # scripted | idle | random

    @property
    def is_two_slot(self) -> bool:
        """Whether this contract uses the two-slot (economy + tactical) layout."""

        return int(self.action_slots) == 2

    @property
    def action_nvec(self) -> tuple[int, ...]:
        """Per-head action cardinalities, derived from the single source of truth.

        Two layouts, selected by :attr:`action_slots`:

        * ``1`` -- ``[kind, group, x, y, target]``, the historical single slot;
        * ``2`` -- ``[econ_* x5, tac_* x5]``, economy slot first, exactly the
          order :func:`ironzero.rwpp.decoding.multi_action_nvec` returns.

        The ``kind`` head must stay in lock-step with ``mock_engine.ACTION_KINDS``
        (which defines the action id order) *and* with the Java bridge dispatch.
        ``ACTION_KINDS`` is imported lazily to avoid a circular import, and the
        two-slot branch imports its own vocabulary for the same reason --
        ``rwpp.decoding`` imports this module, so a top-level import would close
        the cycle.
        """

        if int(self.action_slots) == 2:
            from .rwpp.decoding import multi_action_nvec

            return multi_action_nvec(self)

        from .mock_engine import ACTION_KINDS

        return (
            len(ACTION_KINDS),  # kind: index into ACTION_KINDS
            self.groups + 1,  # per-group control + one broadcast slot
            self.position_grid,
            self.position_grid,
            self.target_slots,
        )

    @property
    def flat_obs_dim(self) -> int:
        entity_block = self.max_entities * self.entity_features
        grid_block = self.map_channels * self.map_grid * self.map_grid
        type_block = 2 * self.max_entities * self.type_features
        return self.global_dim + 2 * entity_block + grid_block + type_block

    @property
    def type_block_offset(self) -> int:
        """Start offset of the per-entity identity block (after the control grid)."""

        return (
            self.global_dim
            + 2 * self.max_entities * self.entity_features
            + self.map_channels * self.map_grid * self.map_grid
        )

    @property
    def terrain_channel(self) -> bool:
        """True when the observation carries the static terrain passability channel."""

        return self.map_channels > 3 and bool(self.terrain_path)

    def grid_to_world(self, gx: int, gy: int) -> tuple[float, float]:
        return (
            _grid_index_to_world(gx, self.position_grid, self.map_width, 24.0),
            _grid_index_to_world(gy, self.position_grid, self.map_height, 24.0),
        )


DEFAULT_CONFIG = MicroDuelConfig()

#: What the engine charges for each producible unit, in credits.
#:
#: The engine does **not** report these: ``factories[].can_produce_price`` comes
#: back empty on this build, so any Python-side affordability decision has to use
#: a table. This is that table, kept here (rather than in ``rwpp/env.py``, where
#: it started) because the action mask needs it too and importing the env from
#: ``observation/`` would be a cycle.
#:
#: These are the values measured from the engine, not guesses: the extractor
#: upgrade price was read out of ``SpecialAction.getPriceValue()`` in round 19 and
#: came back 1400 where the config said 1200, so the same mistake was assumed here
#: until measured.
UNIT_PRICES: dict[str, int] = {
    "builder": 500,
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

#: Price assumed for a unit name that is not in :data:`UNIT_PRICES`.
#:
#: Deliberately the *middle* of the table rather than the cheapest: an unknown
#: custom unit is more likely to be expensive, and under-estimating its price is
#: what produces ``PRODUCE_UNAVAILABLE`` rejections.
UNKNOWN_UNIT_PRICE = 500

#: Enemy visibility budget the **real-game training and evaluation paths** use.
#:
#: ``MicroDuelConfig``'s own default of ``8`` is the historical layout and is
#: deliberately left alone (every shipped checkpoint and the mock engine's
#: fixtures are expressed in it). It is also too small to play on: a 1200-frame
#: Lake replay (``runs/platform/real-rwpp-20261116-133006-d97c5a``) dropped 26.1%
#: of enemy unit-frames and kept 61.7% buildings, so ``c_artillery`` -- the
#: longest-range land unit -- was visible on 0 of the 500 frames it existed for.
#:
#: ``32`` covers the population actually observed: that replay peaks at 16, and
#: ``runs/platform/twoslot-mask-long/eval_report.json`` records ``enemy_alive``
#: peaks of 29-41 on the same map. ``target_slots`` is ``+1`` for the "no target"
#: fallback the target head needs to stay samplable.
#:
#: Keeping this beside ``DEFAULT_CONFIG`` (rather than editing it) is what lets a
#: run opt back into the narrow layout for an A/B without the library contract
#: moving underneath the checkpoints.
WIDE_MAX_ENTITIES: int = 32
WIDE_TARGET_SLOTS: int = WIDE_MAX_ENTITIES + 1
