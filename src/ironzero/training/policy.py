# SPDX-License-Identifier: Apache-2.0
"""Masked hierarchical actor-critic policy for the micro-duel environment.

Action layout (see ``MicroDuelConfig.action_nvec``)::

    [kind(len(ACTION_KINDS)), group(G), x(P), y(P), target(T)]

Heads are sampled hierarchically: the kind is sampled first, then only the
heads that are meaningful for that kind contribute to the log-probability and
entropy. ``stop``/``hold`` therefore do not drag unused position and target
heads into the policy gradient.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

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

MASK_KEYS: tuple[str, ...] = ("kind", "group", "x", "y", "target")


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
        for index, key in enumerate(MASK_KEYS):
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

    def _active_matrix(self, kinds: torch.Tensor, device: torch.device) -> torch.Tensor:
        """(B, num_heads) boolean matrix marking heads active for each kind."""

        from ..mock_engine import ACTION_KINDS

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
        active = self._active_matrix(actions[:, 0], actions.device)
        return (per_head * active).sum(dim=1)

    def entropy(
        self,
        dists: list[torch.distributions.Categorical],
        actions: torch.Tensor,
    ) -> torch.Tensor:
        per_head = torch.stack([dist.entropy() for dist in dists], dim=1)
        active = self._active_matrix(actions[:, 0], actions.device)
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
) -> dict[str, torch.Tensor]:
    """Stack a list of per-env mask dicts into batched boolean tensors."""

    result: dict[str, list[torch.Tensor]] = {key: [] for key in MASK_KEYS}
    for mask in masks:
        for key in MASK_KEYS:
            result[key].append(torch.as_tensor(mask[key], dtype=torch.bool, device=device))
    return {key: torch.stack(values) for key, values in result.items()}
