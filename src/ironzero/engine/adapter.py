# SPDX-License-Identifier: Apache-2.0
"""MicroDuelAdapter: the game-side adapter interface used by the env and bridge."""

from __future__ import annotations

from typing import Any, Iterable

from ..micro_config import MicroDuelConfig
from ..terrain import TerrainGrid
from .engine import MicroDuelEngine


class MicroDuelAdapter:
    """Game-side adapter interface shared by the env and the TCP bridge."""

    def __init__(
        self,
        config: MicroDuelConfig | None = None,
        opponent: str | None = None,
        terrain: TerrainGrid | None = None,
    ) -> None:
        self.engine = MicroDuelEngine(config=config, opponent=opponent, terrain=terrain)
        self.terrain = terrain
        self.opponent = self.engine.opponent
        self.episode_id: str = "ep-0"

    def reset(self, *, episode_id: str, seed: int, scenario: str = "micro_duel") -> dict[str, Any]:
        if scenario != "micro_duel":
            raise ValueError(f"unsupported scenario: {scenario}")
        self.episode_id = episode_id
        self.engine.reset(seed=seed)
        return self.engine.observation()

    def step(self, actions: Iterable[dict[str, Any]], *, team: int = 0) -> dict[str, Any]:
        return self.engine.step_decision(list(actions), team=team)

    def evaluate(self, team: int = 0) -> dict[str, Any]:
        return self.engine.evaluate(team)

    def close(self) -> None:  # pragma: no cover - symmetry with TCP adapters
        return None
