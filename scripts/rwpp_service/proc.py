# SPDX-License-Identifier: Apache-2.0
"""受管子进程执行与任务终止。

``_run_managed_proc`` 是唯一的游戏进程启动点：它把 Popen 登记进
``ACTIVE_PROCESSES``，从而让 ``stop_job`` / ``stop_all_jobs`` 能立刻 kill 掉
正在跑的一局，而不用等 ``communicate`` 的 timeout。

``_episode_timeout`` 由 tick 数与 pacing 反推单局墙钟预算（原先硬编码 300 s
会把 20000 tick 的长局误判为 timeout）。
"""

from __future__ import annotations

import subprocess
import time

from .paths import REPO_ROOT
from .recover import kill_pid_tree
from .state import (
    ACTIVE_PROCESSES,
    EPISODE_TIMEOUT_MARGIN_S,
    JOBS,
    JOBS_LOCK,
    PPO_PROCESSES,
    PPO_STOP,
    STOPPED_JOBS,
    TICK_OVERHEAD_MS,
)

__all__ = ["_episode_timeout", "_run_managed_proc", "stop_all_jobs", "stop_job"]


def _run_managed_proc(
    job_id: str, command: list[str], timeout: float
) -> tuple[int, str]:
    """Execute a subprocess while tracking it so stop_job can terminate it immediately."""

    if job_id in STOPPED_JOBS:
        return -1, ""
    try:
        proc = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except Exception as exc:
        return -1, str(exc)

    with JOBS_LOCK:
        ACTIVE_PROCESSES.setdefault(job_id, []).append(proc)

    try:
        stdout, _ = proc.communicate(timeout=timeout)
        return proc.returncode, stdout or ""
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except OSError:
            pass
        stdout, _ = proc.communicate()
        return -1, stdout or ""
    except Exception:
        try:
            proc.kill()
        except OSError:
            pass
        return -1, ""
    finally:
        with JOBS_LOCK:
            procs = ACTIVE_PROCESSES.get(job_id, [])
            if proc in procs:
                procs.remove(proc)


def stop_job(job_id: str) -> bool:
    """Safely terminate all subprocesses associated with a job and mark it stopped."""

    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return False
        STOPPED_JOBS.add(job_id)
        job["status"] = "stopped"
        job["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        procs = list(ACTIVE_PROCESSES.get(job_id, []))
        if job_id in PPO_PROCESSES:
            procs.append(PPO_PROCESSES[job_id])
            PPO_STOP.add(job_id)
        adopted_pid = int(job.get("pid") or 0) if job.get("adopted") else 0
        for p in procs:
            try:
                p.kill()
            except OSError:
                pass
        ACTIVE_PROCESSES[job_id] = []
    if adopted_pid:
        # An adopted job has no local Popen handle (it outlived a restart), so the
        # pid tree is killed instead. Also tell its monitor thread to wind down.
        PPO_STOP.add(job_id)
        kill_pid_tree(adopted_pid)
    return True



def stop_all_jobs() -> int:
    with JOBS_LOCK:
        running = [jid for jid, j in JOBS.items() if j.get("status") == "running"]
    for jid in running:
        stop_job(jid)
    return len(running)


def _episode_timeout(ticks: int, frame_ms: int) -> float:
    """Wall-clock budget for one episode, derived from its tick count and pacing.

    The old hard-coded 300 s cut every long request short (20000 ticks at the
    real-time 16 ms budget needs 330 s+ of pacing alone), so the episode was
    always reported as ``timeout``. Deriving the budget keeps long runs honest
    while still bounding a wedged game.
    """

    per_tick_ms = max(0, int(frame_ms)) + TICK_OVERHEAD_MS
    return max(120.0, round(ticks * per_tick_ms / 1000.0 + EPISODE_TIMEOUT_MARGIN_S, 1))
