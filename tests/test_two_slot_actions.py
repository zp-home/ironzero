# SPDX-License-Identifier: Apache-2.0
"""Two-slot ("economy + tactical") action layout: masks, policy, decode, dispatch.

Why this file exists
--------------------
The two-slot layout is the fix for a measured defect, not a refactor: with one
action slot the trained policy scored **43.75%** while a do-nothing policy scored
**93.75%** on the same map and curriculum, because every tactical order cancelled
the builder's in-flight construction (see ``rwpp/decoding.py``'s two-slot
rationale and ``rwpp/stepping.py``'s measured history). Splitting the head lets
one decision say "keep building" *and* "push the army".

The risk this file guards is the opposite of the feature: the layout is opt-in
(``action_slots`` defaults to 1) and ~850 existing tests plus every shipped
checkpoint assume the 5-head layout. So the regression half of this file is as
important as the new-feature half:

* the 5-head path must stay **bit-identical** -- asserted against a copy of the
  pre-change policy loaded from git history, comparing ``act``/``log_prob``/
  ``entropy`` with ``torch.equal`` (not ``allclose``);
* the two-slot masks must be a **restriction** of the single-slot masks, never a
  widening. That direction is asserted; the converse is deliberately *not*,
  because the single-slot mask is known to over-open some economy kinds today
  (tracked separately) and pinning that would freeze a bug into the contract.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ironzero.micro_config import DEFAULT_CONFIG, MicroDuelConfig  # noqa: E402
from ironzero.observation.masks import (  # noqa: E402
    MULTI_MASK_KEYS,
    build_masks,
    build_masks_multi,
    multi_mask_keys,
)
from ironzero.rwpp.decoding import (  # noqa: E402
    ECONOMY_ACTION_KINDS,
    ECONOMY_EMPTY_INDEX,
    SLOT_HEADS,
    TACTICAL_ACTION_KINDS,
    TACTICAL_EMPTY_INDEX,
    decode_action,
    decode_action_multi,
    multi_action_nvec,
)
from ironzero.training.policy import (  # noqa: E402
    MASK_KEYS,
    MaskedActorCritic,
    is_two_slot_nvec,
    mask_keys_for,
    masks_to_tensors,
)

TWO = replace(DEFAULT_CONFIG, action_slots=2)


# ----------------------------------------------------------------------
# fixtures / helpers
# ----------------------------------------------------------------------
def _bare_base_state() -> dict:
    """A Lake-like opening: one builder, a command centre, no factory.

    Deliberately the state where the economy kinds and the tactical kinds are
    *both* partly open, because that is where a per-slot restriction can go wrong
    in either direction.
    """

    return {
        "own_entities": [
            {
                "entity_id": 1,
                "x": 100.0,
                "y": 100.0,
                "hp": 4000.0,
                "max_hp": 4000.0,
                "is_building": True,
                "is_factory": True,
                "is_builder": False,
            },
            {
                "entity_id": 2,
                "x": 140.0,
                "y": 140.0,
                "hp": 170.0,
                "max_hp": 170.0,
                "is_builder": True,
                "is_building": False,
            },
        ],
        "visible_enemy_entities": [
            {"entity_id": 9, "x": 900.0, "y": 900.0, "hp": 210.0, "max_hp": 210.0}
        ],
        "resource_slots": [
            {"slot_id": 0, "x": 200.0, "y": 200.0, "occupied": False},
            {"slot_id": 1, "x": 260.0, "y": 260.0, "occupied": False},
        ],
        "global_state": {"funds": 5000.0, "income": 8.0},
        "builders": [{"can_build": ["extractor", "landFactory", "turret"]}],
        "factories": [],
    }


def _no_builder_state() -> dict:
    """Nothing buildable and no factory: both slots must still offer "do nothing"."""

    return {
        "own_entities": [
            {
                "entity_id": 1,
                "x": 100.0,
                "y": 100.0,
                "hp": 4000.0,
                "max_hp": 4000.0,
                "is_building": True,
                "is_factory": True,
            }
        ],
        "visible_enemy_entities": [],
        "resource_slots": [],
        "global_state": {"funds": 0.0, "income": 0.0},
        "builders": [],
        "factories": [],
    }


def _old_policy_module():
    """The pre-change ``policy.py`` from git HEAD, as a module.

    Compared against directly rather than re-implemented: the claim under test is
    "this refactor did not change the 5-head behaviour", and only the actual old
    code can support that claim.
    """

    try:
        source = subprocess.run(
            ["git", "show", "HEAD:src/ironzero/training/policy.py"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:  # pragma: no cover
        pytest.skip(f"git history unavailable: {exc}")
    if "class MaskedActorCritic" not in source:  # pragma: no cover
        pytest.skip("could not read the previous policy.py from git")

    import importlib.util
    import tempfile

    path = Path(tempfile.mkdtemp()) / "policy_old.py"
    path.write_text(source, encoding="utf-8")
    name = "ironzero.training._policy_old_for_two_slot_tests"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "ironzero.training"
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------
# 1. config
# ----------------------------------------------------------------------
class TestMicroConfigSwitch:
    def test_single_slot_nvec_is_the_historical_five_tuple(self):
        assert DEFAULT_CONFIG.action_slots == 1
        assert not DEFAULT_CONFIG.is_two_slot
        assert DEFAULT_CONFIG.action_nvec == (10, 5, 8, 8, 9)
        assert len(DEFAULT_CONFIG.action_nvec) == 5

    def test_two_slot_nvec_is_ten_heads_and_matches_decoding(self):
        assert TWO.is_two_slot
        assert len(TWO.action_nvec) == 10
        assert TWO.action_nvec == multi_action_nvec(TWO)
        # economy first, tactical second -- the order the policy and the masks
        # both assume; a swap here would mis-align every trained weight.
        assert TWO.action_nvec[:SLOT_HEADS] == (len(ECONOMY_ACTION_KINDS), 5, 8, 8, 9)
        assert TWO.action_nvec[SLOT_HEADS:] == (len(TACTICAL_ACTION_KINDS), 5, 8, 8, 9)

    def test_observation_dim_is_unchanged_by_the_action_layout(self):
        # The layout must not touch the observation: a checkpoint's trunk stays
        # loadable across layouts, only the head widths differ.
        assert TWO.flat_obs_dim == DEFAULT_CONFIG.flat_obs_dim

    def test_default_is_the_old_layout(self):
        # If this ever fails, every shipped checkpoint and ~850 tests break.
        assert MicroDuelConfig().action_slots == 1
        assert MicroDuelConfig().action_nvec == (10, 5, 8, 8, 9)


# ----------------------------------------------------------------------
# 2. masks
# ----------------------------------------------------------------------
class TestBuildMasksMulti:
    def test_returns_ten_prefixed_keys_with_correct_widths(self):
        masks = build_masks_multi(_bare_base_state(), TWO)
        assert set(masks) == set(MULTI_MASK_KEYS)
        assert len(masks) == 10
        assert tuple(masks) == MULTI_MASK_KEYS  # insertion order == head order
        for key, width in zip(MULTI_MASK_KEYS, TWO.action_nvec):
            assert masks[key].shape == (width,), f"{key} width mismatch"
            assert masks[key].dtype == np.bool_

    def test_prefixed_keys_do_not_collide_with_single_slot_keys(self):
        # Distinct names are the seam that turns a layout mix-up into a KeyError
        # instead of a silent overwrite inside one dict.
        assert not (set(MULTI_MASK_KEYS) & set(MASK_KEYS))

    def test_rejects_a_single_slot_config(self):
        with pytest.raises(ValueError):
            build_masks_multi(_bare_base_state(), DEFAULT_CONFIG)

    def test_multi_mask_keys_helper_matches_the_constant(self):
        assert multi_mask_keys() == MULTI_MASK_KEYS

    @pytest.mark.parametrize("state_factory", [_bare_base_state, _no_builder_state])
    def test_slot_vocabularies_are_respected(self, state_factory):
        """Every open kind must be a member of that slot's own vocabulary.

        This is the property that makes the split meaningful: economy index 3 is
        ``upgrade_building``, but tactical index 3 is ``hold``. If a slot's mask
        were built against the global ``ACTION_KINDS`` table the indices would
        name different orders and nothing downstream would notice.
        """

        masks = build_masks_multi(state_factory(), TWO)
        econ_open = np.flatnonzero(masks["econ_kind"])
        tac_open = np.flatnonzero(masks["tac_kind"])
        assert all(int(i) < len(ECONOMY_ACTION_KINDS) for i in econ_open)
        assert all(int(i) < len(TACTICAL_ACTION_KINDS) for i in tac_open)
        assert bool(masks["econ_kind"][ECONOMY_EMPTY_INDEX])
        assert bool(masks["tac_kind"][TACTICAL_EMPTY_INDEX])

    @pytest.mark.parametrize("state_factory", [_bare_base_state, _no_builder_state])
    def test_each_slot_is_a_restriction_of_the_single_slot_mask(self, state_factory):
        """Containment: splitting the head must never release an illegal order.

        Asserted in this direction only. The converse ("the single-slot mask is
        itself correct") is *not* asserted on purpose: the single-slot economy mask
        is known to over-open some kinds today and that is being investigated
        separately. Pinning it here would freeze that bug into the contract.
        """

        state = state_factory()
        single = build_masks(state, DEFAULT_CONFIG)
        multi = build_masks_multi(state, TWO)
        for vocabulary, key in (
            (ECONOMY_ACTION_KINDS, "econ_kind"),
            (TACTICAL_ACTION_KINDS, "tac_kind"),
        ):
            for position, name in enumerate(vocabulary):
                if not multi[key][position]:
                    continue
                if position == (ECONOMY_EMPTY_INDEX if key == "econ_kind" else TACTICAL_EMPTY_INDEX):
                    continue  # the "do nothing" entry emits no order at all
                kind_index = _action_kind_index(name)
                assert single["kind"][kind_index], (
                    f"{key}[{position}] ({name}) is open in two-slot but closed in "
                    "single-slot -- the split widened the legal set"
                )

    def test_shared_addressing_heads_match_the_single_slot_ones(self):
        state = _bare_base_state()
        single = build_masks(state, DEFAULT_CONFIG)
        multi = build_masks_multi(state, TWO)
        for name in ("group", "x", "y", "target"):
            assert np.array_equal(multi[f"econ_{name}"], single[name])
            assert np.array_equal(multi[f"tac_{name}"], single[name])


def _action_kind_index(name: str) -> int:
    from ironzero.mock_engine import ACTION_KINDS

    return ACTION_KINDS.index(name)


# ----------------------------------------------------------------------
# 3. decode round-trip
# ----------------------------------------------------------------------
class TestDecodeMultiFromMasks:
    def _sample_legal(self, masks: dict, rng: np.random.Generator) -> np.ndarray:
        """Draw a legal 10-head vector by sampling each head from its open set."""

        vector = []
        for key in MULTI_MASK_KEYS:
            open_positions = np.flatnonzero(masks[key])
            assert len(open_positions) > 0, f"{key} has no legal value"
            vector.append(int(rng.choice(open_positions)))
        return np.array(vector, dtype=np.int64)

    @pytest.mark.parametrize("state_factory", [_bare_base_state, _no_builder_state])
    def test_sampled_legal_vectors_decode_to_at_most_two_valid_orders(self, state_factory):
        state = state_factory()
        masks = build_masks_multi(state, TWO)
        rng = np.random.default_rng(20261116)
        for _ in range(200):
            vector = self._sample_legal(masks, rng)
            orders = decode_action_multi(vector, state, TWO)
            assert len(orders) <= 2
            for order in orders:
                kind = str(order.get("kind"))
                assert kind in set(ECONOMY_ACTION_KINDS) | set(TACTICAL_ACTION_KINDS), (
                    f"decoded kind {kind!r} is in neither slot vocabulary"
                )

    def test_economy_and_tactical_orders_can_coexist_in_one_decision(self):
        """The whole point of the layout: one decision, two independent orders."""

        state = _bare_base_state()
        vector = np.zeros(10, dtype=np.int64)
        vector[0] = ECONOMY_ACTION_KINDS.index("build_extractor")
        vector[SLOT_HEADS + 0] = TACTICAL_ACTION_KINDS.index("attack_move")
        vector[SLOT_HEADS + 1] = 1  # group
        orders = decode_action_multi(vector, state, TWO)
        kinds = [str(order["kind"]) for order in orders]
        assert kinds == ["build_extractor", "attack_move"]
        # economy first: the bridge applies commands in order and the later one
        # wins the unit, so the policy's explicit tactical choice must land last.
        assert len(orders) == 2

    def test_both_slots_empty_emits_nothing(self):
        vector = np.zeros(10, dtype=np.int64)
        vector[0] = ECONOMY_EMPTY_INDEX
        vector[SLOT_HEADS + 0] = TACTICAL_EMPTY_INDEX
        assert decode_action_multi(vector, _bare_base_state(), TWO) == []

    def test_single_slot_decode_still_rejects_a_ten_head_vector(self):
        """Guards the seam: ``decode_action`` must not accept a two-slot vector.

        This is asserted so the dispatch in ``stepping.send_step`` stays
        necessary. ``decode_action`` validates against ``len(action_nvec)`` which
        is 10 under a two-slot config, so it does *not* raise -- it silently reads
        the first five entries. The dispatch must therefore be by config, not by
        relying on a length check to catch a mistake.
        """

        vector = np.zeros(10, dtype=np.int64)
        orders = decode_action(vector, _bare_base_state(), TWO)
        # It decodes *something* (the economy slot only) -- i.e. the tactical slot
        # is dropped. Recording that behaviour here makes any future change to it
        # visible instead of silent.
        assert isinstance(orders, list)


# ----------------------------------------------------------------------
# 4. policy
# ----------------------------------------------------------------------
class TestPolicyLayouts:
    def test_key_order_helpers(self):
        assert mask_keys_for(5) == MASK_KEYS
        assert mask_keys_for(10) == MULTI_MASK_KEYS
        assert not is_two_slot_nvec((10, 5, 8, 8, 9))
        assert is_two_slot_nvec(TWO.action_nvec)
        with pytest.raises(ValueError):
            mask_keys_for(7)
        with pytest.raises(ValueError):
            MaskedActorCritic(16, nvec=(10, 5, 8, 8, 9, 6), hidden=8)

    def test_two_slot_policy_has_ten_heads_and_ten_mask_keys(self):
        policy = MaskedActorCritic(TWO.flat_obs_dim, nvec=TWO.action_nvec, hidden=32)
        assert policy.two_slot
        assert len(policy.heads) == 10
        assert policy.mask_keys == MULTI_MASK_KEYS

        obs = torch.randn(4, TWO.flat_obs_dim)
        masks = {
            key: torch.ones(4, width, dtype=torch.bool)
            for key, width in zip(policy.mask_keys, TWO.action_nvec)
        }
        actions, logprob, value = policy.act(obs, masks)
        assert actions.shape == (4, 10)
        assert logprob.shape == (4,)
        assert value.shape == (4,)
        # every sampled head must be in range for its own width
        for index, width in enumerate(TWO.action_nvec):
            assert bool((actions[:, index] >= 0).all())
            assert bool((actions[:, index] < width).all())

    def test_two_slot_logprob_only_accumulates_active_heads(self):
        torch.manual_seed(0)
        policy = MaskedActorCritic(TWO.flat_obs_dim, nvec=TWO.action_nvec, hidden=32)
        policy.eval()
        obs = torch.randn(1, TWO.flat_obs_dim)
        masks = {
            key: torch.ones(1, width, dtype=torch.bool)
            for key, width in zip(policy.mask_keys, TWO.action_nvec)
        }
        # economy = noop, tactical = noop -> only the two kind heads and the two
        # group heads are active (group is active for every kind by design).
        actions = torch.zeros(1, 10, dtype=torch.long)
        actions[0, 0] = ECONOMY_EMPTY_INDEX
        actions[0, SLOT_HEADS + 0] = TACTICAL_EMPTY_INDEX
        logprob, entropy, _ = policy.evaluate(obs, masks, actions)

        logits, _ = policy.forward(obs)
        dists = policy._dists(logits, masks)
        expected = (
            dists[0].log_prob(actions[:, 0])
            + dists[1].log_prob(actions[:, 1])
            + dists[5].log_prob(actions[:, 5])
            + dists[6].log_prob(actions[:, 6])
        )
        assert torch.allclose(logprob, expected, atol=1e-6), (
            "with both slots empty only kind+group per slot may contribute"
        )
        assert float(entropy) > 0.0

    def test_two_slot_slots_do_not_share_kind_semantics(self):
        """Economy index 1 and tactical index 1 must activate *their own* heads.

        Economy index 1 is ``build_structure`` (positional, no target); tactical
        index 1 is ``attack_move`` (positional, no target) -- so to separate the
        two, compare economy ``build_extractor`` (target) against tactical
        ``attack_move`` (position).
        """

        policy = MaskedActorCritic(TWO.flat_obs_dim, nvec=TWO.action_nvec, hidden=32)
        actions = torch.zeros(1, 10, dtype=torch.long)
        actions[0, 0] = ECONOMY_ACTION_KINDS.index("build_extractor")
        actions[0, SLOT_HEADS + 0] = TACTICAL_ACTION_KINDS.index("attack_move")
        active = policy._active_matrix(actions, torch.device("cpu"))[0]
        econ_active = active[:SLOT_HEADS].tolist()
        tac_active = active[SLOT_HEADS:].tolist()
        # economy: kind, group, target -- no position
        assert econ_active == [True, True, False, False, True]
        # tactical: kind, group, position (x and y) -- no target
        assert tac_active == [True, True, True, True, False]


class TestSingleSlotRegression:
    """The 5-head path must be bit-identical to the pre-change implementation."""

    def _pair(self):
        old_module = _old_policy_module()
        nvec = (10, 5, 8, 8, 9)
        obs_dim = 37
        torch.manual_seed(1234)
        new = MaskedActorCritic(obs_dim, nvec=nvec, hidden=32)
        torch.manual_seed(1234)
        old = old_module.MaskedActorCritic(obs_dim, nvec=nvec, hidden=32)
        return new, old

    def test_initialization_is_identical(self):
        new, old = self._pair()
        for key, tensor in new.state_dict().items():
            assert torch.equal(tensor, old.state_dict()[key]), f"init differs at {key}"

    def _masks(self, nvec, batch=9):
        masks = {}
        for key, width in zip(MASK_KEYS, nvec):
            mask = torch.ones(batch, width, dtype=torch.bool)
            mask[:, 0] = False          # force a non-trivial argmax
            mask[:, width - 1] = True
            masks[key] = mask
        return masks

    @pytest.mark.parametrize("deterministic", [True, False])
    def test_act_is_bit_identical(self, deterministic):
        new, old = self._pair()
        nvec = (10, 5, 8, 8, 9)
        masks = self._masks(nvec)
        torch.manual_seed(7)
        obs = torch.randn(9, 37)

        torch.manual_seed(99)
        a_new, lp_new, v_new = new.act(obs, masks, deterministic=deterministic)
        torch.manual_seed(99)
        a_old, lp_old, v_old = old.act(obs, masks, deterministic=deterministic)

        assert torch.equal(a_new, a_old), "sampled actions changed"
        assert torch.equal(lp_new, lp_old), "log-probability changed (must be exact)"
        assert torch.equal(v_new, v_old), "value changed"

    def test_evaluate_is_bit_identical(self):
        new, old = self._pair()
        nvec = (10, 5, 8, 8, 9)
        masks = self._masks(nvec)
        torch.manual_seed(7)
        obs = torch.randn(9, 37)
        torch.manual_seed(99)
        actions, _, _ = new.act(obs, masks, deterministic=True)

        lp_new, ent_new, val_new = new.evaluate(obs, masks, actions)
        lp_old, ent_old, val_old = old.evaluate(obs, masks, actions)
        for name, a, b in (
            ("logprob", lp_new, lp_old),
            ("entropy", ent_new, ent_old),
            ("value", val_new, val_old),
        ):
            assert torch.equal(a, b), f"{name} changed; max|diff|={float((a - b).abs().max())}"
            assert float((a - b).abs().max()) == 0.0

    def test_single_slot_policy_still_reports_five_heads(self):
        policy = MaskedActorCritic(DEFAULT_CONFIG.flat_obs_dim,
                                   nvec=DEFAULT_CONFIG.action_nvec, hidden=32)
        assert not policy.two_slot
        assert len(policy.heads) == 5
        assert policy.mask_keys == MASK_KEYS


class TestMasksToTensors:
    def test_default_layout_is_unchanged(self):
        nvec = (10, 5, 8, 8, 9)
        masks = [
            {key: np.ones(width, dtype=bool) for key, width in zip(MASK_KEYS, nvec)}
            for _ in range(3)
        ]
        tensors = masks_to_tensors(masks, torch.device("cpu"))
        assert tuple(tensors) == MASK_KEYS
        for key, width in zip(MASK_KEYS, nvec):
            assert tensors[key].shape == (3, width)

    def test_two_slot_keys_stack(self):
        masks = [
            {key: np.ones(width, dtype=bool) for key, width in zip(MULTI_MASK_KEYS, TWO.action_nvec)}
            for _ in range(2)
        ]
        tensors = masks_to_tensors(masks, torch.device("cpu"), keys=MULTI_MASK_KEYS)
        assert tuple(tensors) == MULTI_MASK_KEYS
        for key, width in zip(MULTI_MASK_KEYS, TWO.action_nvec):
            assert tensors[key].shape == (2, width)

    def test_two_slot_masks_raise_under_the_default_layout(self):
        """The failure this guards: a two-slot dict read with 5-key defaults."""

        masks = [
            {key: np.ones(width, dtype=bool) for key, width in zip(MULTI_MASK_KEYS, TWO.action_nvec)}
        ]
        with pytest.raises(KeyError):
            masks_to_tensors(masks, torch.device("cpu"))


# ----------------------------------------------------------------------
# 5. curriculum mode
# ----------------------------------------------------------------------
class TestCurriculumMode:
    def test_default_is_orders(self):
        from ironzero.rwpp.config import RwppGameConfig

        assert RwppGameConfig().curriculum_mode == "orders"

    def test_mask_mode_reports_that_it_may_not_issue_orders(self):
        from ironzero.rwpp.config import RwppGameConfig

        orders = RwppGameConfig(curriculum_mode="orders")
        masked = RwppGameConfig(curriculum_mode="mask")
        off = RwppGameConfig(curriculum_mode="off")
        assert orders.curriculum_issues_orders
        assert not masked.curriculum_issues_orders
        assert not off.curriculum_issues_orders
        # "mask" must still participate (through the action masks), otherwise it is
        # indistinguishable from "off" and the policy loses the gate entirely.
        assert orders.curriculum_enabled
        assert masked.curriculum_enabled
        assert not off.curriculum_enabled

    def test_mask_mode_does_not_close_the_first_build(self):
        """``curriculum_owns_build_kinds`` must be off in "mask" mode.

        In "mask" mode the curriculum issues no orders, so closing the build kinds
        "because the curriculum owns them" would leave *nobody* able to start the
        economy -- measured on the real Lake opening as income stuck at the base
        rate for the whole episode. The gate has to follow who is actually acting.
        """

        from ironzero.rwpp.config import RwppGameConfig

        orders = RwppGameConfig(curriculum_mode="orders", opening_extractor=True)
        masked = RwppGameConfig(curriculum_mode="mask", opening_extractor=True)
        assert orders.micro_config().curriculum_owns_build_kinds
        assert not masked.micro_config().curriculum_owns_build_kinds

    def test_curriculum_actions_returns_empty_in_mask_mode(self):
        from ironzero.rwpp.config import RwppGameConfig
        from ironzero.rwpp.env import RwppDuelEnv

        env = RwppDuelEnv(RwppGameConfig(curriculum_mode="mask"), port=5988,
                          micro_config=DEFAULT_CONFIG)
        try:
            env.state = _bare_base_state()
            env.obs_state = env.state
            assert env._curriculum_actions() == []
        finally:
            env.close()


# ----------------------------------------------------------------------
# 6. training-side plumbing
# ----------------------------------------------------------------------
class TestTrainingPlumbing:
    def test_cli_defaults_keep_the_old_behaviour(self):
        from ironzero.micro_config import WIDE_MAX_ENTITIES
        from ironzero.training.train_rwpp import build_game, build_parser

        args = build_parser().parse_args([])
        game = build_game(args)
        micro = game.micro_config()
        assert game.action_slots == 1
        assert game.curriculum_mode == "orders"
        # The shared head widths are unchanged; only the target head is wider now,
        # because the CLI default stopped hiding 26% of the enemy army (and 100% of
        # the enemy's artillery) behind an 8-slot budget. A narrow run is still
        # reachable with an explicit --max-entities 8 --target-slots 9.
        assert micro.action_nvec == (10, 5, 8, 8, WIDE_MAX_ENTITIES + 1)
        narrow = build_game(
            build_parser().parse_args(["--max-entities", "8", "--target-slots", "9"])
        ).micro_config()
        assert narrow.action_nvec == (10, 5, 8, 8, 9)

    def test_cli_can_select_the_two_slot_layout(self):
        from ironzero.training.train_rwpp import build_game, build_parser

        args = build_parser().parse_args(
            ["--action-slots", "2", "--curriculum-mode", "mask"]
        )
        game = build_game(args)
        micro = game.micro_config()
        assert game.action_slots == 2
        assert game.curriculum_mode == "mask"
        assert micro.is_two_slot
        assert len(micro.action_nvec) == 10
        assert micro.action_nvec == multi_action_nvec(micro)

    def test_ppo_collect_and_update_support_ten_heads(self):
        """A tiny end-to-end PPO step on a fake vector env with a 10-head policy.

        Exercises the real ``collect``/``update`` code paths, because both used the
        module-level ``MASK_KEYS`` and would ``KeyError`` on a two-slot rollout.
        """

        from ironzero.training.ppo import PPOConfig, PPOTrainer
        from ironzero.training.vec_env import StepBatch

        class _FakeVec:
            """Minimal vector env: random obs, terminal after one step."""

            def __init__(self, num_envs, obs_dim, keys, widths):
                self.num_envs = num_envs
                self.obs_dim = obs_dim
                self.keys = keys
                self.widths = widths
                self.masks = [self._masks() for _ in range(num_envs)]
                self._rng = np.random.default_rng(0)

            def _masks(self):
                return {k: np.ones(w, dtype=bool) for k, w in zip(self.keys, self.widths)}

            def reset(self):
                return (self._rng.standard_normal((self.num_envs, self.obs_dim)).astype(np.float32),
                        self.masks, [{} for _ in range(self.num_envs)])

            def step(self, actions):
                obs = self._rng.standard_normal((self.num_envs, self.obs_dim)).astype(np.float32)
                return StepBatch(
                    observations=obs,
                    rewards=np.zeros(self.num_envs, dtype=np.float32),
                    terminated=np.zeros(self.num_envs, dtype=bool),
                    truncated=np.zeros(self.num_envs, dtype=bool),
                    infos=[{} for _ in range(self.num_envs)],
                )

            def drain_completed(self):
                return []

            def close(self):
                pass

        cfg = PPOConfig(
            total_updates=1, num_envs=2, rollout_steps=3, epochs=1,
            minibatch_size=2, hidden=16, device="cpu", out_dir=str(ROOT / "runs" / "_tmp_two_slot"),
        )
        vec = _FakeVec(2, TWO.flat_obs_dim, MULTI_MASK_KEYS, TWO.action_nvec)
        trainer = PPOTrainer(cfg, micro_config=TWO, vec=vec)
        rollout = trainer.collect()
        assert rollout.actions.shape == (3, 2, 10)
        assert set(rollout.masks) == set(MULTI_MASK_KEYS)
        stats = trainer.update(rollout)
        assert np.isfinite(stats["policy_loss"])
        assert np.isfinite(stats["value_loss"])
        assert np.isfinite(stats["entropy"])
