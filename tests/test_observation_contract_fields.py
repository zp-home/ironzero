# SPDX-License-Identifier: Apache-2.0
"""The observation contract must carry the facts a decision in this genre needs.

Every assertion here corresponds to information that was *measurably absent* from
the real bridge payload on ``[p2]Lake (2p)``, not to a hypothetical feature. The
measurements live in ``.runtime/stream_dump.json`` and
``.runtime/contract_gap.json``; this module pins the fixes so they cannot silently
regress.

Why a Java-side contract needs Python-side tests at all: the bridge is compiled
code talking to a decompiled game engine, so a wrong method call does not raise --
it returns a plausible number. Two of the bugs fixed here had that exact shape:

* ``unit.placeOnlyOnResPool()`` returns a constant ``false`` on every stock unit,
  so every resource pool reported ``occupied: false`` forever, and the encoder's
  economy feature was structurally dead;
* a stray ``]`` in ``factoriesJson`` made the whole observation line invalid JSON
  while the Python reader was lenient enough to hide it.

Both were found by reading the wire, so both get a wire-level test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ironzero.micro_config import DEFAULT_CONFIG
from ironzero.observation.encoding import encode_observation
from ironzero.observation.features import (
    ATTACK_RANGE_SCALE,
    MOVE_SPEED_SCALE,
    QUEUE_SCALE,
    SHOOT_DELAY_SCALE,
    _entity_type_features,
)

BRIDGE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "ironzero_bridge"
    / "java"
    / "ironzero"
    / "IronZeroBridge.java"
)


def _bridge_source() -> str:
    return BRIDGE.read_text(encoding="utf-8")


#: Identity-block column indices, mirroring ``features._entity_type_features``.
COLUMNS = {
    "is_building": 1,
    "is_factory": 2,
    "is_builder": 3,
    "is_air_unit": 4,
    "build_progress": 5,
    "type_hash": 6,
    "can_attack": 7,
    "attack_range": 8,
    "move_speed": 9,
    "tech_level": 10,
    "can_attack_air": 11,
    "is_defence": 12,
    "shield": 13,
    "shoot_delay": 14,
    "queue_length": 15,
    "queue_progress": 16,
}


def _identity(unit: dict) -> np.ndarray:
    return _entity_type_features(dict(unit))


# ----------------------------------------------------------------------
# The Java side: every named field must actually be emitted, and the JSON it
# builds must be well formed.
# ----------------------------------------------------------------------


def test_entity_json_emits_every_contract_field() -> None:
    """A field the bridge does not emit is a column fed a constant."""

    source = _bridge_source()
    start = source.index("private String entityJson(Unit unit)")
    end = source.index("private int stableTypeId(Unit unit)")
    body = source[start:end]
    for field in (
        "order_type",
        "move_speed",
        "tech_level",
        "shoot_delay",
        "can_attack_air",
        "can_attack_land",
        "shield",
        "max_shield",
        "is_turret",
        "is_laser",
        "is_defence",
        "queue_length",
        "queue_progress",
        "attack_range",
        "can_attack",
    ):
        assert f'\\"{field}\\"' in body, (
            f"entityJson must emit {field!r}; without it the policy is blind to it"
        )


def test_factories_json_reports_the_build_queue() -> None:
    """The queue is what behaviour cloning had no labels for."""

    source = _bridge_source()
    start = source.index("private String factoriesJson()")
    end = source.index("private String entityJson(Unit unit)")
    body = source[start:end]
    for field in ("queued", "queue_length", "queue_progress", "unit_type", "count"):
        assert f'\\"{field}\\"' in body, f"factoriesJson must report {field!r}"
    assert "getQueuedItems()" in body, "the queue must be read from the engine"
    assert "BuildQueueItem" in body, "queue items carry the type and progress"


def test_factories_json_emits_balanced_json() -> None:
    """Regression: a stray ``]`` made every observation line unparseable.

    The Python reader accepted it (it never validated the line), so the bug was
    invisible from inside the env; a strict ``json.loads`` rejected 100% of frames.
    Pinning the exact append sequence is the cheapest way to keep the braces
    balanced without launching a game.
    """

    source = _bridge_source()
    start = source.index("private String factoriesJson()")
    end = source.index("private String entityJson(Unit unit)")
    body = source[start:end]
    # The object is closed by a bare "}" *after* the queued array was closed by
    # "]". Emitting "]}", "]}" would add a second array close.
    assert 'out.append("]}");' not in body, (
        'factoriesJson must not emit a second "]}" after closing the queued array'
    )
    assert "out.append('}');" in body


def test_resource_slot_occupancy_asks_the_unit_type_not_the_instance() -> None:
    """The bug that made every pool permanently ``occupied: false``.

    ``Unit.placeOnlyOnResPool()`` returns a hard-coded ``false`` and no stock
    subclass overrides it; the extractor's ``true`` lives on ``UnitType$1``. Asking
    the instance therefore reported every pool as free, so the policy could not see
    its own extractors and kept re-ordering the same pool.
    """

    source = _bridge_source()
    helper = source[source.index("private boolean placeOnlyOnResPool(Unit unit)") :]
    helper = helper[: helper.index("}") + 1]
    assert "getUnitTypeInterface().placeOnlyOnResPool()" in helper, (
        "occupancy must come from the unit *type*, which is where the engine "
        "actually stores it"
    )
    # No call site may bypass the helper and ask the instance again.
    instance_calls = [
        line
        for line in source.splitlines()
        if "unit.placeOnlyOnResPool()" in line and "private boolean" not in line
    ]
    assert not instance_calls, (
        f"these still ask the instance and will always answer 'false': {instance_calls}"
    )


def test_win_mode_is_parsed_in_its_own_branch() -> None:
    """``--win-mode objective`` must reach the engine.

    A refactor that added ``--seed`` merged the win-mode parsing into the seed
    branch, so the flag read the seed's value as a mode name and logged
    ``unknown --win-mode: 11 (keeping native)``. On a stock skirmish map the
    engine's own objective never fires, so victory became unreachable again --
    the exact blocker the flag was added to fix.
    """

    source = _bridge_source()
    assert '"--win-mode".equals(argument)' in source, (
        "--win-mode must have its own parsing branch"
    )
    assert '"objective".equals(requested)' in source, (
        "objective is the mode that makes stock maps winnable and must be accepted"
    )


def test_turret_tier_uses_the_turret_accessor() -> None:
    """A gun T3 tower reports tier 1 through ``Unit.getTechLevel()``.

    ``Turret`` does not override the base method, so it inherits the constant 1;
    the real tier is ``Turret.getTurretLevel()``. A turret's tier is exactly what
    decides whether a push will work.
    """

    source = _bridge_source()
    start = source.index("private int techLevel(Unit unit)")
    body = source[start : start + 900]
    assert "getTurretLevel()" in body, (
        "turret tiers must come from Turret.getTurretLevel(), not the inherited 1"
    )


def test_laser_defence_counts_as_a_defence_structure() -> None:
    """``LaserDefence.canAttack()`` is ``false``; it intercepts projectiles.

    ``isBuilding() && canAttack()`` therefore classifies the game's most prominent
    defensive building as a harmless farm.
    """

    source = _bridge_source()
    start = source.index("private String entityJson(Unit unit)")
    end = source.index("private int stableTypeId(Unit unit)")
    body = source[start:end]
    assert "isLaserType(unit)" in body
    assert "canAttack(unit) || isLaserType(unit)" in body, (
        "is_defence must include laser towers, whose canAttack() is false"
    )


def test_queue_fields_cast_to_the_interface_not_the_class() -> None:
    """``CustomUnit`` implements ``FactoryInterface`` without extending ``Factory``.

    Casting to ``Factory`` would report "no queue" for every modded producer.
    """

    source = _bridge_source()
    for method in ("private int queueLength(Unit unit)", "private float queueProgress(Unit unit)"):
        start = source.index(method)
        body = source[start : start + 1200]
        assert "instanceof com.corrodinggames.rts.game.units.buildings.FactoryInterface" in body, (
            f"{method} must test FactoryInterface, which covers custom producers too"
        )


def test_can_upgrade_requires_a_finished_building() -> None:
    """``getBuildActionId()`` means "this type has a next tier", not "upgrade now".

    A structure that is still being placed reports a real upgrade action for its
    whole construction, so ``can_upgrade`` was ``true`` from the first tick of the
    opening extractor. The Python mask opens ``upgrade_building`` whenever *any*
    own entity reports ``can_upgrade``, so for the entire build the mask advertised
    an order the engine always refused.

    Measured on ``[p2]Lake (2p)`` (``.runtime/raw_entity.json``): a half-built
    extractor reported ``can_upgrade: true`` at ``build_progress`` 0.000, 0.029,
    0.060 ... for 18000 ticks. Over 24 PPO updates that produced
    ``UPGRADE_UNAVAILABLE = 798`` against ``BUILD_UNAVAILABLE = 157``, i.e. most of
    ``invalid = -17.49`` -- the dominant negative reward term in every run.
    """

    source = _bridge_source()
    start = source.index("private boolean canUpgrade(Unit unit)")
    body = source[start : start + 1600]
    assert "isNonOperational()" in body, (
        "canUpgrade must require a finished building, or the mask advertises "
        "upgrades the engine refuses for the whole construction"
    )


def test_can_upgrade_gate_precedes_the_action_id_lookup() -> None:
    """Order matters: the completion check must short-circuit first."""

    source = _bridge_source()
    start = source.index("private boolean canUpgrade(Unit unit)")
    body = source[start : start + 1600]
    assert body.index("isNonOperational()") < body.index("getBuildActionId()"), (
        "the completion check must come before the action-id lookup"
    )


# ----------------------------------------------------------------------
# The Python side: the columns must carry the values, in range.
# ----------------------------------------------------------------------


def test_identity_stride_matches_the_declared_type_features() -> None:
    feats = _identity({"is_building": True})
    assert feats.shape == (DEFAULT_CONFIG.type_features,)
    assert DEFAULT_CONFIG.type_features == 17


def test_movement_speed_column_separates_static_from_mobile() -> None:
    tower = _identity({"is_building": True, "is_defence": True, "move_speed": 0.0})
    tank = _identity({"move_speed": 1.0})
    assert tower[COLUMNS["move_speed"]] == 0.0
    assert abs(float(tank[COLUMNS["move_speed"]]) - (1.0 / MOVE_SPEED_SCALE)) < 1e-6


def test_tech_level_is_one_based_and_clamped() -> None:
    t1 = _identity({"tech_level": 1})
    t3 = _identity({"tech_level": 3})
    hostile = _identity({"tech_level": 99})
    assert abs(float(t1[COLUMNS["tech_level"]]) - (1.0 / 3.0)) < 1e-6
    assert abs(float(t3[COLUMNS["tech_level"]]) - 1.0) < 1e-6
    assert float(hostile[COLUMNS["tech_level"]]) <= 1.0
    # An absent ``tech_level`` reads as T1 -- the engine's own default -- and never
    # as "tier zero", a state the game does not have.
    absent = _identity({})
    assert abs(float(absent[COLUMNS["tech_level"]]) - (1.0 / 3.0)) < 1e-6


def test_shield_is_a_fraction_and_survives_a_zero_maximum() -> None:
    half = _identity({"shield": 40.0, "max_shield": 80.0})
    none = _identity({})
    assert abs(float(half[COLUMNS["shield"]]) - 0.5) < 1e-6
    assert none[COLUMNS["shield"]] == 0.0
    assert np.isfinite(none).all()


def test_shoot_delay_and_queue_columns_use_their_scales() -> None:
    busy = _identity({"shoot_delay": 90.0, "queue_length": 2, "queue_progress": 0.5})
    assert abs(float(busy[COLUMNS["shoot_delay"]]) - (90.0 / SHOOT_DELAY_SCALE)) < 1e-6
    assert abs(float(busy[COLUMNS["queue_length"]]) - (2 / QUEUE_SCALE)) < 1e-6
    assert abs(float(busy[COLUMNS["queue_progress"]]) - 0.5) < 1e-6


def test_queue_length_keeps_small_counts_distinguishable() -> None:
    """One vs two queued items must not both saturate or both round to 0."""

    one = _identity({"queue_length": 1})
    two = _identity({"queue_length": 2})
    assert float(one[COLUMNS["queue_length"]]) > 0.0
    assert float(two[COLUMNS["queue_length"]]) > float(one[COLUMNS["queue_length"]])


def test_every_column_stays_in_the_unit_interval_for_hostile_input() -> None:
    hostile = {
        "move_speed": 1e9,
        "tech_level": 10**6,
        "shoot_delay": 1e9,
        "shield": 1e9,
        "max_shield": 1e-9,
        "queue_length": 1e9,
        "queue_progress": 99.0,
        "attack_range": 1e9,
        "build_progress": -5.0,
    }
    feats = _identity(hostile)
    assert np.isfinite(feats).all(), "a hostile value must never produce NaN/inf"
    assert float(feats.min()) >= 0.0
    assert float(feats.max()) <= 1.0


def test_encoder_reads_the_new_fields_end_to_end() -> None:
    """A payload carrying only the new fields must change the encoded vector.

    Guards the plumbing, not just the feature function: if
    ``observation_to_state`` started dropping the keys, this fails.
    """

    def state(unit: dict) -> dict:
        return {
            "tick": 0,
            "own_entities": [dict(unit)],
            "visible_enemy_entities": [],
            "global_state": {"funds": 500.0, "income": 0.0},
            "resource_slots": [],
        }

    base = {"entity_id": 5, "entity_type_id": 1, "x": 100.0, "y": 100.0,
            "hp": 100.0, "max_hp": 100.0}
    plain = encode_observation(state(base), DEFAULT_CONFIG)
    rich = encode_observation(
        state(
            dict(
                base,
                move_speed=1.0,
                tech_level=2,
                can_attack_air=True,
                is_defence=True,
                shield=10.0,
                max_shield=20.0,
                shoot_delay=75.0,
                queue_length=3,
                queue_progress=0.25,
            )
        ),
        DEFAULT_CONFIG,
    )
    assert plain.shape == rich.shape == (DEFAULT_CONFIG.flat_obs_dim,)
    assert not np.array_equal(plain, rich), (
        "the new payload fields did not reach the encoded vector"
    )
    # The geometric and global blocks are untouched by identity-block fields.
    assert np.array_equal(plain[: DEFAULT_CONFIG.type_block_offset],
                          rich[: DEFAULT_CONFIG.type_block_offset])


def test_order_names_cover_the_engine_vocabulary() -> None:
    """``build``/``repair`` were missing, so a building builder read as ``idle``."""

    from ironzero.engine.constants import ORDER_IDS, ORDER_NAMES

    for name in ("idle", "move", "attack_move", "attack_target", "hold", "build", "repair"):
        assert name in ORDER_IDS, f"ORDER_NAMES must cover {name!r}"
    # Appending must not have shifted the original indices.
    assert ORDER_NAMES[0] == "idle"
    assert ORDER_NAMES[1] == "move"
    assert ORDER_NAMES[2] == "attack_move"
    assert ORDER_NAMES[3] == "attack_target"
    assert ORDER_NAMES[4] == "hold"
    assert ORDER_NAMES[5] == "build"
    assert ORDER_NAMES[6] == "repair"


def test_non_default_type_features_still_encodes() -> None:
    """A caller may override the stride; the encoder must follow the config."""

    from dataclasses import replace

    cfg = replace(DEFAULT_CONFIG, type_features=17)
    state = {
        "tick": 0,
        "own_entities": [{"entity_id": 1, "entity_type_id": 1, "x": 1.0, "y": 1.0,
                          "hp": 10.0, "max_hp": 10.0, "move_speed": 1.0}],
        "visible_enemy_entities": [],
        "global_state": {},
        "resource_slots": [],
    }
    vector = encode_observation(state, cfg)
    assert vector.shape == (cfg.flat_obs_dim,)


def test_legacy_nine_column_payload_still_encodes() -> None:
    """An older bridge that sends none of the new fields must still work."""

    legacy = {
        "entity_id": 7,
        "entity_type_id": 1007741,
        "x": 1000.0,
        "y": 2000.0,
        "hp": 4000.0,
        "max_hp": 4000.0,
        "is_building": True,
        "is_factory": True,
        "is_builder": False,
        "is_air_unit": False,
        "build_progress": 1.0,
        "can_attack": True,
        "attack_range": 280.0,
    }
    feats = _identity(legacy)
    assert feats.shape == (DEFAULT_CONFIG.type_features,)
    assert feats[COLUMNS["can_attack"]] == 1.0
    assert abs(float(feats[COLUMNS["attack_range"]]) - (280.0 / ATTACK_RANGE_SCALE)) < 1e-6
    # Everything the old payload cannot express reads as the safe default.
    assert feats[COLUMNS["move_speed"]] == 0.0
    assert feats[COLUMNS["queue_length"]] == 0.0
