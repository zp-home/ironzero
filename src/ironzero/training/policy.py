# SPDX-License-Identifier: Apache-2.0
"""Masked hierarchical actor-critic policy for the micro-duel environment.

Two action layouts are supported, selected purely by the **length of ``nvec``**:

* **single slot** (5 heads) -- the historical layout::

      [kind(len(ACTION_KINDS)), group(G), x(P), y(P), target(T)]

* **two slots** (10 heads) -- ``nvec`` comes from
  :func:`ironzero.rwpp.decoding.multi_action_nvec`, economy slot first::

      economy:  [len(ECONOMY_ACTION_KINDS),  group, x, y, target]
      tactical: [len(TACTICAL_ACTION_KINDS), group, x, y, target]

The two-slot layout exists so one decision can say *both* "keep building an
extractor" and "push the army", which a single slot cannot express (every real
order cancels the previous one -- see ``rwpp/decoding.py``'s two-slot rationale).
It is an opt-in layout: ``MicroDuelConfig.action_slots`` defaults to 1, so every
existing checkpoint keeps loading and behaving exactly as before.

Heads are sampled hierarchically: the kind is sampled first, then only the heads
that are meaningful for that kind contribute to the log-probability and
entropy. ``stop``/``hold`` therefore do not drag unused position and target
heads into the policy gradient.

Layout selection is by ``len(nvec)`` (5 vs 10) rather than a separate flag on
purpose: a policy built from a checkpoint has only the saved ``nvec`` to go on,
and a flag could disagree with it. A length other than 5 or 10 is rejected
loudly instead of being guessed at.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ..rwpp.decoding import (
    ECONOMY_ACTION_KINDS,
    ECONOMY_EMPTY_INDEX,
    SLOT_HEADS,
    TACTICAL_ACTION_KINDS,
    TACTICAL_EMPTY_INDEX,
)

# Which *semantic* heads each action kind consumes, keyed by the kind name from
# ``mock_engine.ACTION_KINDS``. Head 0 is always the kind itself; head 1 is the
# unit group and is used by every kind. The remaining heads are conditional:
#   * position heads (2=x, 3=y) are only meaningful for orders that name a point
#   * the target head (4) carries the enemy slot *or* the build slot index
# Deriving the map from names (rather than hard-coding indices) keeps the policy
# in lock-step with the action space even if ACTION_KINDS is reordered.
_POSITION_KINDS: frozenset[str] = frozenset(
    {"move_group", "attack_move", "build_structure"}
)
_TARGET_KINDS: frozenset[str] = frozenset(
    {"attack_target", "build_extractor", "produce_unit", "upgrade_building"}
)

#: Mask keys of the single-slot layout. ``MASK_KEYS`` remains this tuple so that
#: every existing importer (``ppo.py``, ``bc_warmstart_rwpp.py``, ...) keeps
#: working unchanged; the two-slot keys live in :data:`MULTI_MASK_KEYS`.
MASK_KEYS: tuple[str, ...] = ("kind", "group", "x", "y", "target")

#: The five per-slot head names, in ``action_nvec`` order.
_SLOT_HEAD_NAMES: tuple[str, ...] = ("kind", "group", "x", "y", "target")

#: Mask keys of the two-slot layout, prefixed per slot and ordered exactly like
#: :func:`ironzero.rwpp.decoding.multi_action_nvec` (economy five first).
#:
#: Prefixed keys are load-bearing, not cosmetic: ``build_masks`` and
#: ``build_masks_multi`` must be able to coexist in the same dict-of-masks world
#: without one layout's ``"kind"`` silently overwriting the other's.
MULTI_MASK_KEYS: tuple[str, ...] = tuple(
    f"econ_{name}" for name in _SLOT_HEAD_NAMES
) + tuple(f"tac_{name}" for name in _SLOT_HEAD_NAMES)

#: ``nvec`` lengths this class understands, and what each one means.
_SINGLE_SLOT_HEADS = 5
_TWO_SLOT_HEADS = 2 * SLOT_HEADS


def mask_keys_for(nvec_len: int) -> tuple[str, ...]:
    """The mask key order for a policy with ``nvec_len`` heads.

    Shared by :class:`MaskedActorCritic` and ``masks_to_tensors`` so the two can
    never disagree about which keys a given layout uses.
    """

    if int(nvec_len) == _TWO_SLOT_HEADS:
        return MULTI_MASK_KEYS
    if int(nvec_len) == _SINGLE_SLOT_HEADS:
        return MASK_KEYS
    raise ValueError(
        f"unsupported action-head count {nvec_len}: expected {_SINGLE_SLOT_HEADS} "
        f"(single slot) or {_TWO_SLOT_HEADS} (two slots)"
    )


def is_two_slot_nvec(nvec: tuple[int, ...] | list[int]) -> bool:
    """Whether ``nvec`` describes the two-slot layout."""

    return len(tuple(nvec)) == _TWO_SLOT_HEADS


class MaskedActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        nvec: tuple[int, ...] | None = None,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        if nvec is None:
            # Derive from the scenario contract instead of hard-coding a stale
            # default: a mismatch here silently builds a policy whose kind head
            # disagrees with the mask width and fails at the first forward pass.
            from ..micro_config import DEFAULT_CONFIG

            nvec = DEFAULT_CONFIG.action_nvec
        self.obs_dim = obs_dim
        self.nvec = tuple(nvec)
        # Layout is derived from the head count, never from a flag: a policy
        # rebuilt from a checkpoint only has the saved ``nvec`` to go on, and a
        # flag could disagree with it. ``mask_keys_for`` raises on any length
        # other than 5 or 10, so a mismatch fails here rather than at the first
        # forward pass with a shape error several frames away.
        self.two_slot = is_two_slot_nvec(self.nvec)
        self.mask_keys = mask_keys_for(len(self.nvec))
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden, n) for n in self.nvec])
        self.critic = nn.Linear(hidden, 1)
        self.apply(_orthogonal_init)
        # Small orthogonal init on the policy heads keeps the initial entropy high.
        for head, n in zip(self.heads, self.nvec):
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------
    def forward(self, obs: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        features = self.trunk(obs)
        logits = [head(features) for head in self.heads]
        value = self.critic(features).squeeze(-1)
        return logits, value

    @staticmethod
    def _masked(logits: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return logits
        return logits.masked_fill(~mask, -1e9)

    def _dists(
        self,
        logits: list[torch.Tensor],
        masks: dict[str, torch.Tensor] | None,
    ) -> list[torch.distributions.Categorical]:
        dists = []
        for index, key in enumerate(self.mask_keys):
            mask = None if masks is None else masks.get(key)
            dists.append(torch.distributions.Categorical(logits=self._masked(logits[index], mask)))
        return dists

    # ------------------------------------------------------------------
    # Acting / evaluating
    # ------------------------------------------------------------------
    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        masks: dict[str, torch.Tensor] | None,
        *, deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        dists = self._dists(logits, masks)
        batch = obs.shape[0]
        actions = torch.zeros(batch, len(self.nvec), dtype=torch.long, device=obs.device)
        for head_index, dist in enumerate(dists):
            if deterministic:
                actions[:, head_index] = torch.argmax(dist.logits, dim=-1)
            else:
                actions[:, head_index] = dist.sample()
        logprob = self.log_prob(dists, actions)
        return actions, logprob, value

    def _slot_active(self, slot_kinds: torch.Tensor, vocabulary: tuple[str, ...],
                     head_count: int) -> torch.Tensor:
        """(B, head_count) active flags for ONE slot, from that slot's kind head.

        ``vocabulary`` is the slot's own kind table, **not** the global
        ``ACTION_KINDS``: economy index 0 is ``build_extractor`` while global index
        0 is ``move_group``. Resolving through the slot vocabulary is what keeps
        the two slots' position/target semantics independent -- sharing the global
        table would make the economy slot's ``build_extractor`` look positional
        and drag the x/y heads into its gradient.

        ``head_count`` is the slot's real head count (``vocabulary`` length for the
        kind head, but the slot's ``nvec`` slice is authoritative and is what the
        caller passes): a slot always has exactly 5 heads (kind/group/x/y/target),
        whose kind head happens to have ``len(vocabulary)`` entries.
        """

        active = torch.zeros(slot_kinds.shape[0], head_count, dtype=torch.bool,
                             device=slot_kinds.device)
        active[:, 0] = True  # the kind head itself
        if head_count < 2:
            return active
        active[:, 1] = True  # every order names a unit group
        kinds_cpu = slot_kinds.detach().to("cpu").tolist()
        uses_position = torch.zeros(slot_kinds.shape[0], dtype=torch.bool,
                                    device=slot_kinds.device)
        uses_target = torch.zeros(slot_kinds.shape[0], dtype=torch.bool,
                                  device=slot_kinds.device)
        for row, kind_index in enumerate(kinds_cpu):
            name = vocabulary[int(kind_index) % len(vocabulary)]
            if name in _POSITION_KINDS:
                uses_position[row] = True
            if name in _TARGET_KINDS:
                uses_target[row] = True
        if head_count > 2:
            active[:, 2] = uses_position
        if head_count > 3:
            active[:, 3] = uses_position
        if head_count > 4:
            active[:, 4] = uses_target
        return active

    def _active_matrix(self, actions: torch.Tensor, device: torch.device) -> torch.Tensor:
        """(B, num_heads) boolean matrix marking heads active for each decision.

        ``actions`` is the **whole** action row, not just the kind head, because the
        two-slot layout has two kind heads and each one governs its own five heads.
        The single-slot path only ever reads ``actions[:, 0]`` and is otherwise
        unchanged, so its output is bit-identical to before.
        """

        if self.two_slot:
            # Each slot is exactly ``SLOT_HEADS`` heads wide; its kind head is the
            # first of those five and indexes that slot's own vocabulary. Building
            # each side at ``SLOT_HEADS`` (not at ``len(vocabulary)``, which differs
            # between the economy's 5 and the tactical's 6) is what makes the two
            # halves concatenate: an earlier version used the vocabulary length as
            # the head count and produced a 5+6=11-wide matrix against 10 logits.
            econ = self._slot_active(
                actions[:, 0], ECONOMY_ACTION_KINDS, SLOT_HEADS
            )
            tac = self._slot_active(
                actions[:, SLOT_HEADS], TACTICAL_ACTION_KINDS, SLOT_HEADS
            )
            return torch.cat([econ, tac], dim=1)

        from ..mock_engine import ACTION_KINDS

        kinds = actions[:, 0]
        active = torch.zeros(kinds.shape[0], len(self.nvec), dtype=torch.bool, device=device)
        active[:, 0] = True  # the kind head itself
        if len(self.nvec) < 2:
            return active
        active[:, 1] = True  # every order names a unit group
        # Translate kind indices into the semantic flags for this batch.
        kinds_cpu = kinds.detach().to("cpu").tolist()
        uses_position = torch.zeros(kinds.shape[0], dtype=torch.bool, device=device)
        uses_target = torch.zeros(kinds.shape[0], dtype=torch.bool, device=device)
        for row, kind_index in enumerate(kinds_cpu):
            name = ACTION_KINDS[int(kind_index) % len(ACTION_KINDS)]
            if name in _POSITION_KINDS:
                uses_position[row] = True
            if name in _TARGET_KINDS:
                uses_target[row] = True
        if len(self.nvec) > 2:
            active[:, 2] = uses_position
        if len(self.nvec) > 3:
            active[:, 3] = uses_position
        if len(self.nvec) > 4:
            active[:, 4] = uses_target
        return active

    def log_prob(
        self,
        dists: list[torch.distributions.Categorical],
        actions: torch.Tensor,
    ) -> torch.Tensor:
        per_head = torch.stack(
            [dist.log_prob(actions[:, index]) for index, dist in enumerate(dists)], dim=1
        )
        active = self._active_matrix(actions, actions.device)
        return (per_head * active).sum(dim=1)

    def entropy(
        self,
        dists: list[torch.distributions.Categorical],
        actions: torch.Tensor,
    ) -> torch.Tensor:
        per_head = torch.stack([dist.entropy() for dist in dists], dim=1)
        active = self._active_matrix(actions, actions.device)
        return (per_head * active).sum(dim=1)

    def evaluate(
        self,
        obs: torch.Tensor,
        masks: dict[str, torch.Tensor] | None,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        dists = self._dists(logits, masks)
        logprob = self.log_prob(dists, actions)
        entropy = self.entropy(dists, actions)
        return logprob, entropy, value


def _orthogonal_init(module: nn.Module, gain: float = 2 ** 0.5) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.zeros_(module.bias)


def masks_to_tensors(
    masks: list[dict[str, Any]],
    device: torch.device,
    keys: tuple[str, ...] | None = None,
) -> dict[str, torch.Tensor]:
    """Stack a list of per-env mask dicts into batched boolean tensors.

    ``keys`` names the layout explicitly. Older callers pass nothing and get
    :data:`MASK_KEYS` (the five single-slot keys), which keeps their behaviour
    bit-identical. Two-slot callers must pass :data:`MULTI_MASK_KEYS` (or the
    policy's own ``mask_keys``), because the two layouts use disjoint key names --
    a two-slot mask dict has no ``"kind"`` entry, so iterating ``MASK_KEYS`` over
    it would raise ``KeyError``.

    Deriving the layout from the dict's own keys was deliberately rejected: a
    dict with *both* layouts present (e.g. an env that publishes single-slot
    masks for legacy consumers alongside two-slot ones) would then silently pick
    one, and the caller's ``nvec`` is the authority, not the dict.
    """

    layout = MASK_KEYS if keys is None else tuple(keys)
    result: dict[str, list[torch.Tensor]] = {key: [] for key in layout}
    for mask in masks:
        for key in layout:
            result[key].append(torch.as_tensor(mask[key], dtype=torch.bool, device=device))
    return {key: torch.stack(values) for key, values in result.items()}
