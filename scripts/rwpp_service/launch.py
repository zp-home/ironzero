# SPDX-License-Identifier: Apache-2.0
"""由 HTTP payload 创建任务：``/infer`` ``/train`` ``/ppo`` 的准入与装配。

``StartMixin`` 混入 :class:`rwpp_service.handler.Handler`，方法体与拆分前
``Handler._start_ppo`` / ``Handler._start`` / ``Handler._ppo_running`` 等价，
只是把字段解析、参数校验、任务记录模板抽成了模块级纯函数。
"""

from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path

from .paths import REPO_ROOT
from .paths import _persist_job
from .ppo import resolve_init_checkpoint, resolve_preset, run_ppo
from .runners import run_infer, run_train
from . import state
from .state import JOBS, JOBS_LOCK, PORT_SLOTS

__all__ = ["StartMixin", "ppo_bounds_error"]

#: ``/ppo`` 的 6 段数值/枚举校验，返回 ``(message, status)`` 或 ``None``。
_PPO_CHECKS = (
    ("frame_ms must be 0..16", lambda v: 0 <= v["frame_ms"] <= 16),
    ("vision_radius must be 0..2000", lambda v: 0.0 <= v["vision_radius"] <= 2000.0),
    ("updates must be 1..500", lambda v: 1 <= v["updates"] <= 500),
    ("envs must be 1..16", lambda v: 1 <= v["envs"] <= 16),
    ("rollout_steps must be 1..512", lambda v: 1 <= v["rollout_steps"] <= 512),
    ("episode_ticks must be 120..216000", lambda v: 120 <= v["episode_ticks"] <= 216000),
)


def ppo_bounds_error(values: dict) -> str | None:
    """按原顺序校验 PPO 数值区间；``device`` 单独校验以保持原始次序。"""

    for message, check in _PPO_CHECKS:
        if not check(values):
            return message
    return None


class StartMixin:
    """任务创建路由的共享实现（依赖 host 类提供 ``_send``）。"""

    def _start_ppo(self, payload: dict) -> None:
        updates = int(payload.get("updates", 4))
        envs = int(payload.get("envs", 2))
        rollout_steps = int(payload.get("rollout_steps", 20))
        episode_ticks = int(payload.get("episode_ticks", 3600))
        device = str(payload.get("device", "auto"))
        map_name = str(payload.get("map_name") or "Artillery")
        map_path = str(payload.get("map_path") or "maps/challenge/Artillery.tmx")
        map_width = float(payload.get("map_width") or 2700.0)
        map_height = float(payload.get("map_height") or 2500.0)
        frame_ms = int(payload.get("frame_ms", 2))
        vision_radius = float(payload.get("vision_radius", 260.0))
        eval_every = int(payload.get("eval_every", 0))
        eval_episodes = int(payload.get("eval_episodes", 1))
        use_llm = bool(payload.get("use_llm", False))
        preset = resolve_preset(payload, map_name, map_path)
        map_name = preset["map_name"]
        map_path = preset["map_path"]
        arena = preset["arena"]
        units_per_team = preset["units_per_team"]
        max_entities = preset["max_entities"]
        target_slots = preset["target_slots"]
        terrain = preset["terrain"]
        win_mode = preset["win_mode"]
        unlock_ai = preset["unlock_ai"]
        init_checkpoint = resolve_init_checkpoint(payload, units_per_team)

        error = ppo_bounds_error(
            {
                "frame_ms": frame_ms,
                "vision_radius": vision_radius,
                "updates": updates,
                "envs": envs,
                "rollout_steps": rollout_steps,
                "episode_ticks": episode_ticks,
            }
        )
        if error:
            self._send({"error": error}, 400)
            return
        if device not in {"auto", "cpu", "cuda"}:
            self._send({"error": "device must be auto/cpu/cuda"}, 400)
            return
        with JOBS_LOCK:
            running_ppo = [
                job for job in JOBS.values() if job.get("kind") == "ppo" and job.get("status") == "running"
            ]
            running_real = [
                job
                for job in JOBS.values()
                if job.get("kind") in ("infer", "train") and job.get("status") == "running"
            ]
        if running_ppo:
            running_id = running_ppo[0]["id"]
            self._send({"error": f"当前已有 PPO 训练任务正在进行 ({running_id})，请先停止它再启动新训练", "running_job_id": running_id}, 409)
            return
        if running_real:
            running_id = running_real[0]["id"]
            self._send({"error": f"当前已有实机任务正在进行 ({running_id})，请先停止它或等待完成", "running_job_id": running_id}, 409)
            return
        job_id = f"ppo-{uuid.uuid4().hex[:8]}"
        with JOBS_LOCK:
            JOBS[job_id] = {
                "id": job_id,
                "kind": "ppo",
                "status": "running",
                "updates_total": updates,
                "updates_done": 0,
                "envs": envs,
                "rollout_steps": rollout_steps,
                "episode_ticks": episode_ticks,
                "device": device,
                "map_name": map_name,
                "map_path": map_path,
                "frame_ms": frame_ms,
                "vision_radius": vision_radius,
                "eval_every": eval_every,
                "eval_episodes": eval_episodes,
                "use_llm": use_llm,
                "arena": arena,
                "units_per_team": units_per_team,
                "max_entities": max_entities,
                "target_slots": target_slots,
                "terrain": terrain,
                "win_mode": win_mode,
                "unlock_ai": unlock_ai,
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "started_ts": time.time(),
                "elapsed": 0.0,
                "latest_update": None,
                "evaluations": [],
                "summary": None,
                "live_file": None,
                "out_dir": None,
                "error": None,
                "last_log": "",
            }
            # Persist immediately: a restart in the next few seconds must still
            # know this job exists (see ``recover.load_persisted_jobs``).
            _persist_job(job_id, dict(JOBS[job_id]))

        def runner() -> None:
            run_ppo(
                job_id,
                updates=updates,
                envs=envs,
                rollout_steps=rollout_steps,
                episode_ticks=episode_ticks,
                device=device,
                map_name=map_name,
                map_path=map_path,
                map_width=map_width,
                map_height=map_height,
                frame_ms=frame_ms,
                vision_radius=vision_radius,
                eval_every=eval_every,
                eval_episodes=eval_episodes,
                use_llm=use_llm,
                units_per_team=units_per_team,
                max_entities=max_entities,
                target_slots=target_slots,
                terrain=terrain,
                win_mode=win_mode,
                unlock_ai=unlock_ai,
                init_checkpoint=init_checkpoint,
            )

        threading.Thread(target=runner, daemon=True).start()
        self._send({"job_id": job_id, "status": "running", "kind": "ppo"})

    def _start(self, kind: str, payload: dict) -> None:
        ticks = int(payload.get("ticks", 3600))
        policy = str(payload.get("policy", "scripted"))
        if policy not in ("scripted", "none", "model", "ppo"):
            self._send({"error": "policy must be scripted|none|model"}, 400)
            return
        checkpoint = payload.get("checkpoint")
        map_name = payload.get("map_name")
        map_path = payload.get("map_path")
        map_width = float(payload.get("map_width") or 2600.0)
        map_height = float(payload.get("map_height") or 2600.0)
        frame_ms = int(payload.get("frame_ms", 2))
        episodes = int(payload.get("episodes", 1))
        concurrency = int(payload.get("concurrency", 1))
        if kind == "train" and not 1 <= episodes <= 50:
            self._send({"error": "episodes out of range"}, 400)
            return
        if not 1 <= concurrency <= state.MAX_PARALLEL:
            self._send({"error": f"concurrency must be 1..{state.MAX_PARALLEL}"}, 400)
            return
        if not 120 <= ticks <= 216000:
            self._send({"error": "ticks must be 120..216000"}, 400)
            return
        if policy in ("model", "ppo") and not checkpoint:
            self._send({"error": "policy=model requires checkpoint"}, 400)
            return
        if policy in ("model", "ppo") and checkpoint:
            candidate = Path(str(checkpoint))
            if not candidate.is_absolute():
                candidate = REPO_ROOT / candidate
            if not candidate.exists():
                self._send({"error": f"checkpoint not found: {checkpoint}"}, 400)
                return
            checkpoint = str(candidate)
        if self._ppo_running():
            with JOBS_LOCK:
                running_ppo = [job["id"] for job in JOBS.values() if job.get("kind") == "ppo" and job.get("status") == "running"]
            running_id = running_ppo[0] if running_ppo else "unknown"
            self._send({"error": f"当前有 PPO 训练任务正在进行 ({running_id})，请先停止它再启动实机任务", "running_job_id": running_id}, 409)
            return
        if PORT_SLOTS.qsize() <= 0:
            self._send({"error": "all parallel slots are busy"}, 409)
            return
        job_id = f"rwpp-{uuid.uuid4().hex[:8]}"
        with JOBS_LOCK:
            JOBS[job_id] = {
                "id": job_id,
                "kind": kind,
                "status": "running",
                "ticks": ticks,
                "policy": policy,
                "checkpoint": checkpoint,
                "frame_ms": frame_ms,
                "concurrency": concurrency if kind == "train" else 1,
                "episodes_total": episodes if kind == "train" else 1,
                "episodes_done": 0,
                "episodes_metrics": [],
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "started_ts": time.time(),
                "elapsed": 0.0,
                "ticks_total": 0,
                "replays": [],
                "replay_path": None,
                "frames_file": f"runs/platform/rwpp-live-{job_id}/frames.jsonl" if kind == "infer" else None,
                "frames": 0,
                "port": None,
                "map_name": map_name,
                "map_path": map_path,
                "map_width": map_width,
                "map_height": map_height,
            }
            _persist_job(job_id, dict(JOBS[job_id]))

        def runner() -> None:
            if kind == "train":
                run_train(
                    job_id,
                    episodes,
                    ticks,
                    policy,
                    map_name,
                    concurrency,
                    map_path=map_path,
                    map_width=map_width,
                    map_height=map_height,
                )
            else:
                run_infer(
                    job_id,
                    ticks,
                    policy,
                    map_name,
                    map_path=map_path,
                    map_width=map_width,
                    map_height=map_height,
                    checkpoint=checkpoint,
                    frame_ms=frame_ms,
                )

        threading.Thread(target=runner, daemon=True).start()
        self._send({"job_id": job_id, "status": "running", "kind": kind})

    def _ppo_running(self) -> bool:
        with JOBS_LOCK:
            return any(
                job.get("kind") == "ppo" and job.get("status") == "running"
                for job in JOBS.values()
            )
