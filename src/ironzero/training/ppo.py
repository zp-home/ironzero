# SPDX-License-Identifier: Apache-2.0
"""PPO with GAE and per-head action masks for the micro-duel task."""

from __future__ import annotations

import inspect
import json
import threading
import time
import warnings
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from ..micro_config import DEFAULT_CONFIG, MicroDuelConfig
from .evaluate import evaluate_policy, evaluate_random
from .policy import MASK_KEYS, MaskedActorCritic, masks_to_tensors
from .vec_env import VecMicroEnv

@dataclass
class PPOConfig:
    total_updates: int = 120
    num_envs: int = 8
    rollout_steps: int = 128
    epochs: int = 4
    minibatch_size: int = 256
    lr: float = 3e-4
    # ``gamma`` must be read against the *episode length*, not picked by habit.
    # The real-game episode is 3600 simulated ticks / DECISION_INTERVAL(10) =
    # 360 decisions, so the terminal +/-1.0 is discounted by gamma**360:
    #   0.99^360  = 0.027   -> a won game is worth +0.027 in the first step,
    #                          i.e. the policy cannot see victory at all
    #   0.999^360 = 0.698   <- keeps the terminal signal inside the horizon
    # The 0.99 value came from the 120-decision micro-duel task, where
    # 0.99^120 = 0.30 was already borderline. Diagnostics on a 105-update
    # real-game run confirmed the symptom: the terminal/potential/kills terms
    # never entered the return, only the immediately-paid penalties did.
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    hidden: int = 256
    device: str = "auto"
    seed: int = 0
    eval_every: int = 20
    eval_episodes: int = 24
    # Save ``checkpoint_best.pt`` whenever an evaluation beats the previous best
    # (win rate first, mean return second). ``checkpoint.pt`` stays the final
    # update so both "latest" and "best" are always available.
    save_best: bool = True
    async_eval: bool = True
    opponent: str = "scripted"
    out_dir: str = "runs/micro_duel"
    log_every: int = 5

    def resolved_device(self) -> torch.device:
        if self.device != "auto":
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Rollout:
    observations: torch.Tensor  # (T, B, D)
    actions: torch.Tensor  # (T, B, 5)
    logprobs: torch.Tensor  # (T, B)
    values: torch.Tensor  # (T, B)
    rewards: torch.Tensor  # (T, B)
    dones: torch.Tensor  # (T, B)
    masks: dict[str, torch.Tensor]  # (T, B, n)
    last_values: torch.Tensor  # (B,)


class PPOTrainer:
    def __init__(
        self,
        config: PPOConfig | None = None,
        *,
        micro_config: MicroDuelConfig | None = None,
        live_path: str | Path | None = None,
        vec: Any | None = None,
        baseline_fn: Callable[[], dict[str, Any]] | None = None,
        eval_fn: Callable[["PPOTrainer"], dict[str, Any]] | None = None,
    ) -> None:
        self.config = config or PPOConfig()
        self.micro = micro_config or DEFAULT_CONFIG
        self.device = self.config.resolved_device()
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        self.live_path = Path(live_path) if live_path else None
        self._live_handle = None
        self._live_seq = 0
        self._baseline_fn = baseline_fn
        self._eval_fn = eval_fn
        self._collected_strategy: dict[str, Any] | None = None
        self._last_strategy_payload: dict[str, Any] | None = None
        if self.live_path is not None:
            self.live_path.parent.mkdir(parents=True, exist_ok=True)
            self._live_handle = self.live_path.open("a", encoding="utf-8")
        # ``vec`` lets callers plug a different environment family (real game)
        # while reusing the whole PPO/GAE/checkpoint pipeline.
        self.vec = vec if vec is not None else VecMicroEnv(
            self.config.num_envs,
            config=self.micro,
            opponent=self.config.opponent,
            base_seed=self.config.seed,
        )
        self.policy = MaskedActorCritic(
            self.micro.flat_obs_dim,
            nvec=self.micro.action_nvec,
            hidden=self.config.hidden,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.lr, eps=1e-5)
        self.history: list[dict[str, Any]] = []
        self.start_time = time.time()
        self.best_eval: dict[str, Any] | None = None
        self.best_score: tuple[float, float] | None = None
        self.best_update: int | None = None
        self._current_update: int = 0
        self._recent_episodes: deque[dict[str, Any]] = deque(maxlen=20)
        self._eval_thread: threading.Thread | None = None
        self._eval_lock = threading.Lock()
        self._in_flight_eval_update: int | None = None
        # Highest update whose evaluation came due while a worker was busy.
        # Drained on the next opportunity instead of being skipped forever.
        self._pending_eval_update: int | None = None

    # ------------------------------------------------------------------
    # Live event stream (dashboard)
    # ------------------------------------------------------------------
    def _emit(self, kind: str, payload: dict[str, Any]) -> None:
        if self._live_handle is None:
            return
        self._live_seq += 1
        event = {"seq": self._live_seq, "kind": kind, "payload": payload}
        self._live_handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        self._live_handle.flush()

    def close(self) -> None:
        if self._eval_thread is not None and self._eval_thread.is_alive():
            try:
                self._eval_thread.join(timeout=3.0)
            except Exception:
                pass
        if self._live_handle is not None:
            self._live_handle.close()
            self._live_handle = None

    # ------------------------------------------------------------------
    # Baseline / evaluation hooks (offline defaults, real game overrides)
    # ------------------------------------------------------------------
    def _claim_eval_slot(self, update: int) -> int | None:
        """Decide which update to evaluate now, or ``None`` to defer.

        A due evaluation used to be *dropped* whenever the background worker was
        still busy (measured: a 4-episode real-game evaluation took ~112 s while
        the 10 updates it gated took ~15 s, so 3 of every 4 due evaluations never
        ran). Deferring instead means an update that came due is always
        evaluated eventually -- immediately if the worker is free, otherwise as
        soon as it finishes.

        Returns the update number whose weights should be evaluated, or ``None``
        if this call was deferred.
        """

        if self._eval_thread is not None and self._eval_thread.is_alive():
            self._pending_eval_update = max(
                int(self._pending_eval_update or 0), int(update)
            )
            print(
                f"[eval ] update {update:03d} queued: update {self._in_flight_eval_update} "
                f"is still evaluating in background"
            )
            return None
        pending = self._pending_eval_update
        self._pending_eval_update = None
        return int(pending) if pending else int(update)

    def baseline(self) -> dict[str, Any]:
        if self._baseline_fn is not None:
            return self._baseline_fn()
        return evaluate_random(
            episodes=self.config.eval_episodes,
            config=self.micro,
            opponent=self.config.opponent,
        )

    def evaluate(self, update: int, policy: MaskedActorCritic | None = None) -> dict[str, Any]:
        self._current_update = int(update)
        pol = policy if policy is not None else self.policy
        if self._eval_fn is not None:
            sig = inspect.signature(self._eval_fn)
            if "policy" in sig.parameters or len(sig.parameters) >= 2:
                return self._eval_fn(self, policy=pol)
            return self._eval_fn(self)
        return evaluate_policy(
            pol,
            episodes=self.config.eval_episodes,
            config=self.micro,
            opponent=self.config.opponent,
            seed=50_000 + update,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Best-checkpoint bookkeeping
    # ------------------------------------------------------------------
    def _eval_score(self, evaluation: dict[str, Any] | None) -> tuple[float, float]:
        """Ranking key: win rate first (higher), mean return second."""

        if not evaluation:
            return (float("-inf"), float("-inf"))
        try:
            win_rate = float(evaluation.get("win_rate", 0.0))
        except (TypeError, ValueError):
            win_rate = 0.0
        try:
            mean_return = float(evaluation.get("mean_return", 0.0))
        except (TypeError, ValueError):
            mean_return = 0.0
        return (win_rate, mean_return)

    def maybe_save_best(
        self,
        out_dir: Path,
        update: int,
        evaluation: dict[str, Any] | None,
        metrics: dict[str, Any] | None = None,
    ) -> bool:
        """Persist ``checkpoint_best.pt`` when this eval beats the running best."""

        if not self.config.save_best or not evaluation:
            return False
        score = self._eval_score(evaluation)
        if self.best_score is not None and score <= self.best_score:
            return False
        self.best_score = score
        self.best_update = int(update)
        self.best_eval = {"update": int(update), **dict(evaluation)}
        payload: dict[str, Any] = {
            "model": self.policy.state_dict(),
            "micro_config": asdict(self.micro),
            "ppo_config": asdict(self.config),
            "update": int(update),
            "best_eval": self.best_eval,
            "metrics": metrics.get("evaluations", []) if metrics else [],
        }
        torch.save(payload, out_dir / "checkpoint_best.pt")
        return True

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------
    def collect(self) -> Rollout:
        cfg = self.config
        obs, masks, _infos = self.vec.reset()
        observations: list[torch.Tensor] = []
        actions: list[torch.Tensor] = []
        logprobs: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        rewards: list[torch.Tensor] = []
        dones: list[torch.Tensor] = []
        mask_history: dict[str, list[torch.Tensor]] = {key: [] for key in MASK_KEYS}
        for _ in range(cfg.rollout_steps):
            mask_tensors = masks_to_tensors(masks, self.device)
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            action, logprob, value = self.policy.act(obs_tensor, mask_tensors)
            batch = self.vec.step(action.cpu().numpy())
            for info in batch.infos:
                if isinstance(info, dict) and info.get("strategy"):
                    self._collected_strategy = info["strategy"]
            observations.append(obs_tensor.cpu())
            actions.append(action.cpu())
            logprobs.append(logprob.cpu())
            values.append(value.cpu())
            rewards.append(torch.as_tensor(batch.rewards, dtype=torch.float32))
            dones.append(torch.as_tensor(batch.terminated | batch.truncated, dtype=torch.float32))
            for key in MASK_KEYS:
                mask_history[key].append(mask_tensors[key].cpu())
            obs, masks = batch.observations, self.vec.masks
        with torch.no_grad():
            last_obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            last_masks = masks_to_tensors(masks, self.device)
            _, last_values = self.policy.forward(last_obs)
            _ = last_masks
        return Rollout(
            observations=torch.stack(observations),
            actions=torch.stack(actions),
            logprobs=torch.stack(logprobs),
            values=torch.stack(values),
            rewards=torch.stack(rewards),
            dones=torch.stack(dones),
            masks={key: torch.stack(mask_history[key]) for key in MASK_KEYS},
            last_values=last_values.cpu(),
        )

    # ------------------------------------------------------------------
    # Learning update
    # ------------------------------------------------------------------
    def compute_gae(self, rollout: Rollout) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        rewards = rollout.rewards
        values = rollout.values
        dones = rollout.dones
        steps = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros(rewards.shape[1], dtype=torch.float32)
        for step in reversed(range(steps)):
            if step == steps - 1:
                next_value = rollout.last_values
            else:
                next_value = values[step + 1]
            non_terminal = 1.0 - dones[step]
            delta = rewards[step] + cfg.gamma * next_value * non_terminal - values[step]
            last_gae = delta + cfg.gamma * cfg.gae_lambda * non_terminal * last_gae
            advantages[step] = last_gae
        returns = advantages + values
        return returns, advantages

    def update(self, rollout: Rollout) -> dict[str, float]:
        cfg = self.config
        returns, advantages = self.compute_gae(rollout)
        steps, num_envs = advantages.shape
        flat_obs = rollout.observations.reshape(-1, self.micro.flat_obs_dim)
        flat_actions = rollout.actions.reshape(-1, len(self.micro.action_nvec))
        flat_logprobs = rollout.logprobs.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)
        flat_masks = {
            key: value.reshape(steps * num_envs, *value.shape[2:]) for key, value in rollout.masks.items()
        }
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)
        total = flat_obs.shape[0]
        indices = np.arange(total)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_fraction": 0.0, "updates": 0}
        for _epoch in range(cfg.epochs):
            np.random.shuffle(indices)
            for start in range(0, total, cfg.minibatch_size):
                batch = indices[start : start + cfg.minibatch_size]
                obs_batch = flat_obs[batch].to(self.device)
                actions_batch = flat_actions[batch].to(self.device)
                old_logprob = flat_logprobs[batch].to(self.device)
                advantages_batch = flat_advantages[batch].to(self.device)
                returns_batch = flat_returns[batch].to(self.device)
                masks_batch = {key: flat_masks[key][batch].to(self.device) for key in MASK_KEYS}
                new_logprob, entropy, value = self.policy.evaluate(obs_batch, masks_batch, actions_batch)
                ratio = torch.exp(new_logprob - old_logprob)
                surrogate_1 = ratio * advantages_batch
                surrogate_2 = torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * advantages_batch
                policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
                value_loss = 0.5 * ((value - returns_batch) ** 2).mean()
                entropy_mean = entropy.mean()
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_mean
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                stats["policy_loss"] += float(policy_loss.item())
                stats["value_loss"] += float(value_loss.item())
                stats["entropy"] += float(entropy_mean.item())
                stats["clip_fraction"] += float(((ratio - 1.0).abs() > cfg.clip_ratio).float().mean().item())
                stats["updates"] += 1
        divisor = max(1, stats["updates"])
        return {key: value / divisor for key, value in stats.items() if key != "updates"}

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def train(self) -> dict[str, Any]:
        cfg = self.config
        out_dir = Path(cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        random_baseline = self.baseline()
        if random_baseline:
            print(f"[baseline] random: {json.dumps(random_baseline, ensure_ascii=False)}")
        self._emit(
            "start",
            {
                "config": asdict(cfg),
                "device": str(self.device),
                "random_baseline": random_baseline,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
        metrics: dict[str, Any] = {
            "config": asdict(cfg),
            "device": str(self.device),
            "random_baseline": random_baseline,
            "updates": [],
            "evaluations": [],
        }
        for update in range(1, cfg.total_updates + 1):
            rollout = self.collect()
            update_stats = self.update(rollout)
            if self._collected_strategy and self._collected_strategy != self._last_strategy_payload:
                self._last_strategy_payload = self._collected_strategy
                self._emit("strategy", dict(self._collected_strategy))
            completed = self.vec.drain_completed()
            for episode in completed:
                self._recent_episodes.append(episode)
            returns = [episode["return"] for episode in completed]
            results = [episode.get("result") for episode in completed]
            components: dict[str, list[float]] = {}
            for episode in completed:
                for key, value in (episode.get("components") or {}).items():
                    components.setdefault(key, []).append(float(value))
            # Aggregate *why* orders were refused. A non-zero ``invalid`` reward
            # is the symptom; these counts are the diagnosis (e.g. a mask that
            # lets the policy select a group whose units are all dead).
            rejection_totals: dict[str, int] = {}
            for episode in completed:
                for reason, count in (episode.get("rejections") or {}).items():
                    rejection_totals[reason] = (
                        rejection_totals.get(reason, 0) + int(count)
                    )

            recent_returns = [ep["return"] for ep in self._recent_episodes]
            recent_results = [ep.get("result") for ep in self._recent_episodes]
            rolling_return = float(np.mean(recent_returns)) if recent_returns else None
            rolling_win_rate = (
                sum(1 for r in recent_results if r == "victory") / len(recent_results)
                if recent_results
                else None
            )

            # If current update completed episodes, use their exact mean;
            # otherwise smoothly retain the rolling average of recent completed games,
            # avoiding steep drops to 0.0 when no episode ended in this short rollout.
            mean_return = (
                float(np.mean(returns))
                if returns
                else (rolling_return if rolling_return is not None else 0.0)
            )
            win_rate = (
                sum(1 for r in results if r == "victory") / len(results)
                if results
                else (rolling_win_rate if rolling_win_rate is not None else 0.0)
            )
            loss_rate = (
                sum(1 for r in results if r == "defeat") / len(results)
                if results
                else (
                    sum(1 for r in recent_results if r == "defeat") / len(recent_results)
                    if recent_results
                    else 0.0
                )
            )
            timeout_rate = (
                sum(1 for r in results if r == "timeout") / len(results)
                if results
                else (
                    sum(1 for r in recent_results if r == "timeout") / len(recent_results)
                    if recent_results
                    else 0.0
                )
            )

            record = {
                "update": update,
                **update_stats,
                "episodes": len(completed),
                "total_episodes": len(self._recent_episodes),
                "has_completed": bool(self._recent_episodes),
                "mean_return": mean_return,
                "rolling_return": rolling_return if rolling_return is not None else 0.0,
                "win_rate": win_rate,
                "rolling_win_rate": rolling_win_rate if rolling_win_rate is not None else 0.0,
                "loss_rate": loss_rate,
                "timeout_rate": timeout_rate,
                "reward_components": {
                    key: float(np.mean(values)) for key, values in components.items()
                },
                "rejections": rejection_totals,
                "elapsed_s": round(time.time() - self.start_time, 2),
            }
            metrics["updates"].append(record)
            self._emit("update", record)
            if update % cfg.log_every == 0 or update == 1:
                eps_label = (
                    f"{record['episodes']}"
                    if not self._recent_episodes
                    else f"{record['episodes']} (tot={len(self._recent_episodes)})"
                )
                print(
                    "[train] update {update:03d} eps={eps_label} win={win_rate:.2f} loss={loss_rate:.2f} "
                    "timeout={timeout_rate:.2f} return={mean_return:+.3f} "
                    "pi={policy_loss:.3f} v={value_loss:.3f} H={entropy:.3f} t={elapsed_s}s".format(
                        update=update,
                        eps_label=eps_label,
                        mean_return=record["mean_return"],
                        win_rate=record["win_rate"],
                        loss_rate=record["loss_rate"],
                        timeout_rate=record["timeout_rate"],
                        policy_loss=record["policy_loss"],
                        value_loss=record["value_loss"],
                        entropy=record["entropy"],
                        elapsed_s=record["elapsed_s"],
                    )
                )
                if rejection_totals:
                    # Only print when something was actually refused: a clean run
                    # should not grow the log.
                    print(
                        "[train]   rejections: "
                        + " ".join(
                            f"{reason}={count}"
                            for reason, count in sorted(
                                rejection_totals.items(), key=lambda kv: -kv[1]
                            )
                        )
                    )
            # ``eval_every == 0`` means "evaluate only on the final update".
            due = (
                cfg.eval_every > 0 and update % cfg.eval_every == 0
            ) or update == cfg.total_updates
            if due:
                if cfg.async_eval and update < cfg.total_updates:
                    target_update = self._claim_eval_slot(update)
                    if target_update is None:
                        pass  # deferred: a worker is still running
                    else:
                        weights = {k: v.cpu().clone() for k, v in self.policy.state_dict().items()}
                        self._in_flight_eval_update = target_update

                        def _eval_worker(upd: int, state_dict: dict[str, torch.Tensor]) -> None:
                            try:
                                eval_policy = MaskedActorCritic(
                                    self.micro.flat_obs_dim,
                                    nvec=self.micro.action_nvec,
                                    hidden=self.config.hidden,
                                ).to(self.device)
                                eval_policy.load_state_dict(state_dict)
                                eval_policy.eval()

                                evaluation = self.evaluate(upd, policy=eval_policy)
                                evaluations = {"update": upd, **evaluation}
                                with self._eval_lock:
                                    best = self.maybe_save_best(out_dir, upd, evaluation, metrics)
                                    evaluations["is_best"] = bool(best)
                                    metrics["evaluations"].append(evaluations)
                                    self._emit("eval", evaluations)
                                    self.save(out_dir, upd, metrics)
                                if evaluation:
                                    display = {k: v for k, v in evaluation.items() if k != "update"}
                                    print(
                                        f"[eval ] (async) update {upd:03d} win={evaluation.get('win_rate', 0.0):.2f} "
                                        f"loss={evaluation.get('loss_rate', 0.0):.2f} timeout={evaluation.get('timeout_rate', 0.0):.2f} "
                                        f"return={evaluation.get('mean_return', 0.0):+.3f} own_hp={evaluation.get('mean_own_hp_frac', 0.0):.2f}"
                                    )
                            except Exception as exc:
                                print(f"[eval ] async evaluation for update {upd} failed: {exc}")
                            finally:
                                with self._eval_lock:
                                    if self._in_flight_eval_update == upd:
                                        self._in_flight_eval_update = None

                        self._eval_thread = threading.Thread(
                            target=_eval_worker, args=(target_update, weights), daemon=True
                        )
                        self._eval_thread.start()
                else:
                    evaluation = self.evaluate(update)
                    evaluations = {"update": update, **evaluation}
                    with self._eval_lock:
                        best = self.maybe_save_best(out_dir, update, evaluation, metrics)
                        evaluations["is_best"] = bool(best)
                        metrics["evaluations"].append(evaluations)
                        self._emit("eval", evaluations)
                        if evaluation:
                            display = {key: value for key, value in evaluation.items() if key != "update"}
                            print(
                                "[eval ] update {update:03d} win={win_rate:.2f} loss={loss_rate:.2f} "
                                "timeout={timeout_rate:.2f} return={mean_return:+.3f} own_hp={mean_own_hp_frac:.2f}".format(
                                    update=update, **display
                                )
                            )
                        self.save(out_dir, update, metrics)
        if self._eval_thread is not None and self._eval_thread.is_alive():
            print(f"[train] waiting for in-flight background evaluation (update {self._in_flight_eval_update}) to finish...")
            self._eval_thread.join()
        # An evaluation that came due while a worker was busy was deferred, not
        # dropped. Run the last deferred one now so the run reports at least one
        # real measurement for the updates it actually reached.
        deferred = self._pending_eval_update
        self._pending_eval_update = None
        if deferred is not None:
            already = any(
                ev.get("update") == deferred for ev in metrics.get("evaluations", [])
            )
            if not already:
                try:
                    evaluation = self.evaluate(int(deferred))
                    evaluations = {"update": int(deferred), **evaluation}
                    with self._eval_lock:
                        best = self.maybe_save_best(
                            out_dir, int(deferred), evaluation, metrics
                        )
                        evaluations["is_best"] = bool(best)
                        metrics["evaluations"].append(evaluations)
                        self._emit("eval", evaluations)
                        print(
                            f"[eval ] deferred update {int(deferred):03d} "
                            f"win={float(evaluation.get('win_rate', 0.0)):.2f} "
                            f"timeout={float(evaluation.get('timeout_rate', 0.0)):.2f}"
                        )
                except Exception as exc:  # noqa: BLE001
                    print(f"[eval ] deferred evaluation {deferred} failed: {exc}")

        has_final = any(ev.get("update") == cfg.total_updates for ev in metrics.get("evaluations", []))
        if not has_final and (cfg.eval_every > 0 or cfg.total_updates > 0):
            try:
                evaluation = self.evaluate(cfg.total_updates)
                evaluations = {"update": cfg.total_updates, **evaluation}
                best = self.maybe_save_best(out_dir, cfg.total_updates, evaluation, metrics)
                evaluations["is_best"] = bool(best)
                metrics["evaluations"].append(evaluations)
                self._emit("eval", evaluations)
                if evaluation:
                    display = {key: value for key, value in evaluation.items() if key != "update"}
                    print(
                        "[eval ] final update {update:03d} win={win_rate:.2f} loss={loss_rate:.2f} "
                        "timeout={timeout_rate:.2f} return={mean_return:+.3f} own_hp={mean_own_hp_frac:.2f}".format(
                            update=cfg.total_updates, **display
                        )
                    )
            except Exception as exc:
                print(f"[eval ] final evaluation failed safely: {exc}")
        final_eval = metrics["evaluations"][-1] if metrics["evaluations"] else None
        metrics["final_eval"] = final_eval
        metrics["best_eval"] = self.best_eval
        metrics["best_update"] = self.best_update
        metrics["best_checkpoint"] = (
            str(out_dir / "checkpoint_best.pt") if self.best_eval is not None else None
        )
        metrics["total_seconds"] = round(time.time() - self.start_time, 2)
        self.save(out_dir, cfg.total_updates, metrics)
        with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2)
        self._emit(
            "done",
            {
                "final_eval": final_eval,
                "best_eval": self.best_eval,
                "best_update": self.best_update,
                "total_seconds": metrics["total_seconds"],
                "checkpoint": str(out_dir / "checkpoint.pt"),
                "checkpoint_best": metrics["best_checkpoint"],
            },
        )
        self.close()
        return metrics

    # ------------------------------------------------------------------
    def save(self, out_dir: Path, update: int, metrics: dict[str, Any]) -> Path:
        path = out_dir / "checkpoint.pt"
        torch.save(
            {
                "model": self.policy.state_dict(),
                "micro_config": asdict(self.micro),
                "ppo_config": asdict(self.config),
                "update": update,
                "metrics": metrics.get("evaluations", []),
            },
            path,
        )
        best = out_dir / "checkpoint_latest.pt"
        torch.save(
            {
                "model": self.policy.state_dict(),
                "micro_config": asdict(self.micro),
                "ppo_config": asdict(self.config),
                "update": update,
            },
            best,
        )
        return path


def load_policy(path: str | Path, device: str | torch.device = "cpu") -> tuple[MaskedActorCritic, MicroDuelConfig]:
    """Rebuild a policy from a checkpoint.

    The network architecture (observation dim, action heads, hidden width) is
    taken from the checkpoint itself - ``micro_config`` for the task contract
    and ``ppo_config`` for the trunk width - so a checkpoint trained with a
    non-default ``--hidden`` or a 25-vs-25 / terrain observation loads without
    the caller having to know those numbers.
    """

    payload = torch.load(Path(path), map_location=device, weights_only=False)
    raw_micro = dict(payload.get("micro_config") or {})
    ppo_config = dict(payload.get("ppo_config") or {})
    try:
        hidden = int(ppo_config.get("hidden", 256))
    except (TypeError, ValueError):
        hidden = 256
    state_dict = dict(payload["model"])
    trunk_key = "trunk.0.weight"
    if trunk_key in state_dict:
        # Authoritative: the saved matrix shape cannot disagree with the weights.
        hidden = int(state_dict[trunk_key].shape[0])

    # The observation/action contract can grow between releases (the global
    # feature block and the kind head both did). Rebuild against the *current*
    # contract and migrate the weights, instead of rebuilding a stale config
    # that then disagrees with what the live environment produces.
    micro = MicroDuelConfig(**raw_micro)
    target_obs_dim = micro.flat_obs_dim
    target_nvec = micro.action_nvec
    saved_obs_dim = state_dict[trunk_key].shape[1] if trunk_key in state_dict else target_obs_dim

    policy = MaskedActorCritic(target_obs_dim, nvec=target_nvec, hidden=hidden)
    migrated = _migrate_state_dict(policy.state_dict(), state_dict)
    policy.load_state_dict(migrated)
    if saved_obs_dim != target_obs_dim:
        warnings.warn(
            f"checkpoint was trained with obs_dim={saved_obs_dim} but the current "
            f"contract produces {target_obs_dim}; the trunk input was migrated "
            f"(leading weights kept, new features start from zero influence).",
            RuntimeWarning,
            stacklevel=2,
        )
    policy.to(torch.device(device))
    policy.eval()
    return policy, micro


def _migrate_state_dict(target: dict, saved: dict) -> dict:
    """Copy every tensor whose shape still matches; pad/truncate the rest.

    A head or trunk layer whose width changed keeps its shared leading block and
    leaves the new rows/columns at their freshly-initialised values. This makes
    old checkpoints usable after an action-space or observation-space change.
    """

    out = dict(target)
    for name, param in target.items():
        src = saved.get(name)
        if src is None or not hasattr(src, "shape") or not hasattr(param, "shape"):
            continue
        if src.shape == param.shape:
            out[name] = src
            continue
        if src.dim() != param.dim():
            continue
        if src.dim() == 2:
            rows = min(param.shape[0], src.shape[0])
            cols = min(param.shape[1], src.shape[1])
            merged = param.clone()
            merged[:rows, :cols] = src[:rows, :cols]
            out[name] = merged
        elif src.dim() == 1:
            n = min(param.shape[0], src.shape[0])
            merged = param.clone()
            merged[:n] = src[:n]
            out[name] = merged
    return out


__all__ = ["PPOConfig", "PPOTrainer", "Rollout", "load_policy"]
