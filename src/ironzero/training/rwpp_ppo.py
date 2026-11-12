# SPDX-License-Identifier: Apache-2.0
"""PPO training with a policy in the loop of real Rusted Warfare 1.15 games.

This wires :class:`ironzero.training.vec_rwpp.VecRwppEnv` into the existing
PPO/GAE/checkpoint pipeline. Every training step is a real game simulation:
the policy chooses an action, the bridge injects it into the native game, and
the next observation comes from the real engine.
"""

from __future__ import annotations

import threading
from pathlib import Path
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
        strategy: Any | None = None,
    ) -> None:
        self.game = game or RwppGameConfig()
        micro = micro_config or self.game.micro_config()
        self.port_base = int(port_base)
        self.eval_port_base = int(eval_port_base)
        # Background and final evaluation can overlap, so they must not share a
        # bridge port. The pool must stay clear of the training envs' port range
        # (``port_base`` .. ``port_base + num_envs``).
        self.EVAL_PORT_POOL = 16
        self._eval_port_cursor = 0
        self._eval_port_lock = threading.Lock()
        cfg = config or PPOConfig()
        vec = VecRwppEnv(
            cfg.num_envs,
            game=self.game,
            port_base=self.port_base,
            base_seed=cfg.seed,
            micro_config=micro,
            episode_prefix="rwpp-train",
            persistent=True,
            strategy=strategy,
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

    def _real_eval(self, trainer: PPOTrainer, policy: Any | None = None) -> dict[str, Any]:
        out_dir = Path(self.config.out_dir)
        pol = policy if policy is not None else trainer.policy
        # Each evaluation gets its own block of bridge ports, one per parallel
        # episode. Background evaluation (``async_eval``) can still be running
        # when the final evaluation starts -- and both used to bind a single
        # shared port. The second game then died with
        # ``BindException: Address already in use``, the env raised
        # "no observation from real game", and the whole run exited non-zero
        # without writing ``rwpp_summary.json`` (this is exactly how the
        # 2026-11-12 500-update run failed at its final evaluation).
        episodes = max(1, int(self.config.eval_episodes))
        port = self._reserve_eval_ports(episodes)
        return evaluate_rwpp_policy(
            pol,
            game=self.game,
            episodes=episodes,
            micro_config=self.micro,
            port_base=port,
            seed=70_000 + self.config.seed,
            device=trainer.device,
            update=getattr(self, "_current_update", None),
            report_path=out_dir / "eval_report.json",
            parallel=episodes,
        )

    def _reserve_eval_ports(self, count: int) -> int:
        """Reserve ``count`` consecutive evaluation ports and return the first.

        The cursor advances by the whole block, not by one: episodes within one
        evaluation occupy ``first .. first+count-1``, so a +1 rotation would let
        an overlapping background evaluation land inside the block already in
        use by another one.
        """

        count = max(1, int(count))
        with self._eval_port_lock:
            if count > self.EVAL_PORT_POOL:
                raise RuntimeError(
                    f"eval_episodes={count} exceeds the {self.EVAL_PORT_POOL}-port "
                    f"evaluation pool; raise EVAL_PORT_POOL or lower --eval-episodes"
                )
            first = self.eval_port_base + self._eval_port_cursor
            self._eval_port_cursor = (self._eval_port_cursor + count) % self.EVAL_PORT_POOL
            return first

    def close(self) -> None:
        super().close()
        try:
            self.vec.close()
        except Exception:  # noqa: BLE001 - shutdown must never mask training errors
            pass


__all__ = ["RwppPPOTrainer"]
