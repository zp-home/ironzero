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

__all__ = [
    "FAST_FORWARD_MAX",
    "FAST_FORWARD_MIN",
    "LOCKSTEP_TIMEOUT_DEFAULT_MS",
    "_accel_flags",
    "_episode_timeout",
    "_run_managed_proc",
    "stop_all_jobs",
    "stop_job",
]

#: ``--fast-forward`` bounds. ``1`` means "off"; the bridge maps N to the engine's
#: own ``gameSpeed`` (single player / non-replay only), so one tick advances N
#: times the game time and a match needs 1/N the ticks. It *changes fidelity* --
#: unlike ``--frame-ms``, which is pure pacing once ``--physics-ms`` pins the delta.
FAST_FORWARD_MIN = 1
FAST_FORWARD_MAX = 32
#: The bridge's own default for ``--lockstep-timeout-ms``. The flag is only
#: forwarded when a caller asks for something else, so a default command line
#: stays byte-identical to one built before lockstep existed.
LOCKSTEP_TIMEOUT_DEFAULT_MS = 30000


def _accel_flags(
    fast_forward: int = 1,
    lockstep: bool = False,
    lockstep_timeout_ms: int = LOCKSTEP_TIMEOUT_DEFAULT_MS,
) -> list[str]:
    """Bridge acceleration flags for a child command line.

    Everything here is **opt-in**, matching the Python-side contract
    (``ironzero.rwpp.process._bridge_command``): a default request must produce a
    command line that is byte-identical to the pre-``fast_forward`` build, so old
    logs and replays stay comparable. Values are clamped instead of raising --
    a malformed payload is refused by the HTTP layer with a 400 *before* a job
    exists, and a stray flag must never make the bridge reject its whole command.
    """

    flags: list[str] = []
    speed = int(fast_forward or FAST_FORWARD_MIN)
    speed = max(FAST_FORWARD_MIN, min(FAST_FORWARD_MAX, speed))
    if speed != FAST_FORWARD_MIN:
        flags.extend(["--fast-forward", str(speed)])
    if lockstep:
        flags.append("--lockstep")
        timeout_ms = int(lockstep_timeout_ms or LOCKSTEP_TIMEOUT_DEFAULT_MS)
        if timeout_ms != LOCKSTEP_TIMEOUT_DEFAULT_MS:
            flags.extend(["--lockstep-timeout-ms", str(timeout_ms)])
    return flags


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
            # Decode explicitly. ``text=True`` alone uses the locale codec (cp936
            # on a Chinese Windows), and the game/torch write UTF-8: any byte cp936
            # cannot map raised inside ``communicate()``, the blanket ``except``
            # below turned that into ``(-1, "")``, and the job ended as
            # ``status=failed, error=None, last_log=''`` -- an empty report with no
            # way to find out why. ``errors="replace"`` keeps even a genuinely
            # mixed-encoding log readable.
            encoding="utf-8",
            errors="replace",
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


def _episode_timeout(ticks: int, frame_ms: int, fast_forward: int = 1) -> float:
    """Wall-clock budget for one episode, derived from its tick count and pacing.

    The old hard-coded 300 s cut every long request short (20000 ticks at the
    real-time 16 ms budget needs 330 s+ of pacing alone), so the episode was
    always reported as ``timeout``. Deriving the budget keeps long runs honest
    while still bounding a wedged game.

    ``fast_forward`` divides the budget because it shortens the *episode*: with
    ``--fast-forward N`` the engine advances N times the game time per tick, so
    the same match needs 1/N of the ticks. Leaving the budget untouched would
    just mean a stopped/wedged run keeps its slot for N times longer than it
    could possibly need. The 120 s floor is kept -- engine boot, map load and
    the replay write do not get faster with ``gameSpeed``.
    """

    speed = max(1, int(fast_forward or 1))
    per_tick_ms = max(0, int(frame_ms)) + TICK_OVERHEAD_MS
    budget = ticks * per_tick_ms / 1000.0 / speed + EPISODE_TIMEOUT_MARGIN_S
    return max(120.0, round(budget, 1))
