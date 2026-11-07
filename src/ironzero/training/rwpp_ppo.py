# SPDX-License-Identifier: Apache-2.0
"""PPO training with a policy in the loop of real Rusted Warfare 1.15 games.

This wires :class:`ironzero.training.vec_rwpp.VecRwppEnv` into the existing
PPO/GAE/checkpoint pipeline. Every training step is a real game simulation:
the policy chooses an action, the bridge injects it into the native game, and
the next observation comes from the real engine.
"""

from __future__ import annotations

from typing import Any

from ..micro_config import MicroDuelConfig
from ..rwpp_env import RwppGameConfig
from .evaluate_rwpp import evaluate_rwpp_policy
from .ppo import PPOConfig, PPOTrainer
from .vec_rwpp import VecRwppEnv


class RwppPPOTrainer(PPOTrainer):
    """PPO trainer whose environment family is the real game."""

    def __init__(
        self,
        config: PPOConfig | None = None,
        *,
        game: RwppGameConfig | None = None,
        micro_config: MicroDuelConfig | None = None,
        live_path: str | None = None,
        port_base: int = 5140,
        eval_port_base: int = 5200,
    ) -> None:
        self.game = game or RwppGameConfig()
        micro = micro_config or self.game.micro_config()
        self.port_base = int(port_base)
        self.eval_port_base = int(eval_port_base)
        cfg = config or PPOConfig()
        vec = VecRwppEnv(
            cfg.num_envs,
            game=self.game,
            port_base=self.port_base,
            base_seed=cfg.seed,
            micro_config=micro,
            episode_prefix="rwpp-train",
            persistent=True,
        )
        super().__init__(
            cfg,
            micro_config=micro,
            live_path=live_path,
            vec=vec,
            baseline_fn=None,  # a random baseline is not meaningful on the real game
            eval_fn=self._real_eval,
        )

    # ------------------------------------------------------------------
    def baseline(self) -> dict[str, Any]:
        """No offline random baseline: only real-game numbers are reported."""

        return {}

    def _real_eval(self, trainer: PPOTrainer) -> dict[str, Any]:
        return evaluate_rwpp_policy(
            trainer.policy,
            game=self.game,
            episodes=self.config.eval_episodes,
            micro_config=self.micro,
            port_base=self.eval_port_base,
            seed=70_000 + self.config.seed,
            device=trainer.device,
        )

    def close(self) -> None:
        super().close()
        try:
            self.vec.close()
        except Exception:  # noqa: BLE001 - shutdown must never mask training errors
            pass


__all__ = ["RwppPPOTrainer"]
