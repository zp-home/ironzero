# SPDX-License-Identifier: Apache-2.0
"""Masked hierarchical actor-critic policy for the micro-duel environment.

Action layout (see ``MicroDuelConfig.action_nvec``)::

    [kind(5), group(G), x(P), y(P), target(T)]

Heads are sampled hierarchically: the kind is sampled first, then only the
heads that are meaningful for that kind contribute to the log-probability and
entropy. ``stop``/``hold`` therefore do not drag unused position and target
heads into the policy gradient.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

# Heads that are semantically active per action kind:
# 0 move, 1 attack_move, 2 attack_target, 3 stop, 4 hold
ACTIVE_HEADS: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 3),
    (0, 1, 2, 3),
    (0, 1, 4),
    (0, 1),
    (0, 1),
)

MASK_KEYS: tuple[str, ...] = ("kind", "group", "x", "y", "target")


class MaskedActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        nvec: tuple[int, ...] = (5, 4, 8, 8, 9),
        hidden: int = 256,
    ) -> None:
        super().__init__()
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
        """(B, 5) boolean matrix marking heads active for the sampled kind."""

        active = torch.zeros(kinds.shape[0], len(self.nvec), dtype=torch.bool, device=device)
        active[:, 0] = True
        active[:, 1] = True
        active[:, 2] = kinds <= 1
        active[:, 3] = kinds <= 1
        active[:, 4] = kinds == 2
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

    def greedy_action(self, obs: torch.Tensor, masks: dict[str, torch.Tensor] | None) -> torch.Tensor:
        actions, _, _ = self.act(obs, masks, deterministic=True)
        return actions


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
