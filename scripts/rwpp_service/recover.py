# SPDX-License-Identifier: Apache-2.0
"""服务重启后接管仍在运行的任务（adoption）。

``run_ppo`` 的轮询循环是**服务进程内的线程**，而 trainer 是独立子进程（它自己
再拉起一批 JVM）。所以服务一重启，``JOBS`` 表就空了，但机器上其实还在训练：
Web 大屏"什么都看不到"，用户既不能观察也不能停止。这曾经真实发生过——重启
一次服务，正在跑的第 107/500 轮训练就从平台上消失了。

本模块补两件事：

1. ``paths._update`` 会把任务记录落盘到 ``runs/_rwpp_service/jobs/``，重启后
   可以原样恢复 ``JOBS``；
2. 没有落盘记录的（本功能上线前启动的任务）从**磁盘产物**反推：``rwpp-ppo-*``
   目录里有 ``live.jsonl`` 且没有 ``rwpp_summary.json``，且 ``live.jsonl`` 仍在
   更新，就视为在跑，交给 :func:`_monitor_adopted` 继续刷新进度。

终态判定优先看产物，而不是猜：

* ``rwpp_summary.json`` 出现 → ``done``（并读出 summary）；
* ``train.log`` 出现 ``Traceback`` → ``failed``（错误取最后一行）；
* ``live.jsonl`` 超过 :data:`STALE_SECONDS` 没更新且无 summary → ``stale``。
  这里故意**不报 failed**：训练可能只是卡在一个很慢的 update 上，谎报失败比
  诚实地说"没动静了"更糟。

停止：调用方仍走 ``/jobs/{id}/stop``。adopted 任务没有本地 ``Popen`` 句柄，
所以按 pid 用 ``taskkill /T /F`` 整棵杀掉（Windows 零依赖）；pid 未知时
``stop_job`` 返回 ``None``，由 handler 明确告诉用户需要手工停。
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path

from . import state
from .paths import JOB_STATE_DIR, REPO_ROOT, RUNS, _relative, _update
from .state import JOBS, JOBS_LOCK

__all__ = [
    "STALE_SECONDS",
    "adoptable_job_dirs",
    "kill_pid_tree",
    "load_persisted_jobs",
    "recover_jobs",
]

#: ``live.jsonl`` 静默多久算"没动静了"。训练一个 update 可能很慢（5000 tick ×
#: 64 rollout），所以给足余量；宁可晚判 stale，也不要误判在跑的任务已死。
STALE_SECONDS = 300.0
#: 监视线程的轮询间隔（与原 ``run_ppo`` 的 2 s 一致）。
POLL_SECONDS = 2.0


def kill_pid_tree(pid: int) -> bool:
    """Kill ``pid`` and its children. Zero-dependency on Windows (taskkill)."""

    if int(pid) <= 0:
        return False
    try:
        if os.name == "nt":
            result = subprocess.run(
                ["taskkill", "/PID", str(int(pid)), "/T", "/F"],
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        os.kill(int(pid), 9)
        return True
    except OSError:
        return False


def _pid_alive(pid: int) -> bool:
    """Best-effort liveness probe for a pid we did not spawn."""

    pid = int(pid or 0)
    if pid <= 0:
        return False
    try:
        if os.name == "nt":
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True,
                text=True,
            )
            return str(pid) in (result.stdout or "")
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _trainer_pids() -> dict[str, int]:
    """``--out <dir>`` -> pid for the trainers still running, if discoverable.

    ``psutil`` makes this exact; without it we simply cannot attribute a pid to
    an on-disk run, and adoption degrades to read-only monitoring (progress is
    visible, stopping needs a manual kill). Never a hard dependency.
    """

    return _scan_trainers()[0]


def _scan_trainers() -> tuple[dict[str, int], dict[str, list[str]]]:
    """``{(resolved --out): pid}`` and ``{(resolved --out): cmdline}``.

    The command line is the only place the scenario map name survives (the
    trainer's startup summary records ``episode_ticks``/``units_per_team`` but
    not the map), so adopted jobs take ``--map``/``--map-name`` from here.
    """

    try:
        import psutil  # noqa: PLC0415 - optional, host-only convenience
    except Exception:  # noqa: BLE001 - optional dependency
        return {}, {}
    pids: dict[str, int] = {}
    cmdlines: dict[str, list[str]] = {}
    try:
        for proc in psutil.process_iter(["pid", "cmdline"]):
            cmdline = proc.info.get("cmdline") or []
            if "ironzero.training.train_rwpp" not in " ".join(cmdline):
                continue
            for index, token in enumerate(cmdline):
                if token == "--out" and index + 1 < len(cmdline):
                    try:
                        key = str(Path(cmdline[index + 1]).resolve())
                    except (OSError, ValueError):
                        break
                    pids[key] = int(proc.info["pid"])
                    cmdlines[key] = cmdline
                    break
    except Exception:  # noqa: BLE001 - psutil can raise on access denied
        return pids, cmdlines
    return pids, cmdlines


def _flag(cmdline: list[str], name: str) -> str | None:
    """Value of ``--name <value>`` in a command line, or ``None``."""

    for index, token in enumerate(cmdline):
        if token == name and index + 1 < len(cmdline):
            return cmdline[index + 1]
    return None


def _int_or_none(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _float_or_none(value: object) -> float | None:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _read_events(path: Path) -> list[dict]:
    events: list[dict] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        return events
    return events


def _started_ts(start: dict | None, live_path: Path) -> float:
    """Best-effort start timestamp, preferring the ``start`` event's own stamp."""

    created = ((start or {}).get("payload") or {}).get("created_at")
    if created:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                return time.mktime(time.strptime(str(created), fmt))
            except ValueError:
                continue
    try:
        return live_path.stat().st_mtime
    except OSError:
        return time.time()


def _startup_summary(job_dir: Path) -> dict:
    """The trainer's first log line as a dict (map / episode_ticks / units...).

    ``live.jsonl`` only carries the PPO hyper-parameters, but the dashboard also
    shows the scenario (map, ``episode_ticks``, army size). That lives in the
    trainer's first stdout line, which is a single JSON object; parse it rather
    than guess, and return ``{}`` when the log is missing or still empty.
    """

    log_path = job_dir / "train.log"
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            for _ in range(20):
                line = handle.readline()
                if not line:
                    break
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    payload = json.loads(line)
                except ValueError:
                    continue
                if isinstance(payload, dict) and payload.get("mode"):
                    return payload
    except OSError:
        return {}
    return {}


def _job_from_live(job_dir: Path, cmdline: list[str] | None = None) -> dict | None:
    """Rebuild a job record from a run directory (no persisted record needed)."""

    live_path = job_dir / "live.jsonl"
    if not live_path.is_file():
        return None
    events = _read_events(live_path)
    if not events:
        return None
    job_id = job_dir.name[len("rwpp-ppo-") :] if job_dir.name.startswith("rwpp-ppo-") else job_dir.name
    start = next((event for event in events if event.get("kind") == "start"), None)
    config = ((start or {}).get("payload") or {}).get("config") or {}
    updates = [event["payload"] for event in events if event.get("kind") == "update"]
    evaluations = [event["payload"] for event in events if event.get("kind") == "eval"]
    started_ts = _started_ts(start, live_path)
    summary = _startup_summary(job_dir)
    # Scenario: prefer the live process's own flags, fall back to the log summary.
    cmdline = cmdline or []
    map_path = _flag(cmdline, "--map")
    map_name = _flag(cmdline, "--map-name")
    if not map_name and map_path:
        map_name = Path(str(map_path).replace("\\", "/")).stem
    return {
        "id": job_id,
        "kind": "ppo",
        "status": "running",
        "updates_total": int(config.get("total_updates") or 0),
        "updates_done": len(updates),
        "envs": int(config.get("num_envs") or 0),
        "rollout_steps": int(config.get("rollout_steps") or 0),
        "episode_ticks": int(summary.get("episode_ticks") or 0) or None,
        "device": config.get("device") or "",
        "map_name": map_name or summary.get("map"),
        "map_path": map_path or "",
        "frame_ms": _int_or_none(_flag(cmdline, "--frame-ms")) if cmdline else summary.get("frame_ms"),
        "vision_radius": (
            _float_or_none(_flag(cmdline, "--vision-radius")) if cmdline else summary.get("vision_radius")
        ),
        "units_per_team": (
            _int_or_none(_flag(cmdline, "--units-per-team")) if cmdline else summary.get("units_per_team")
        ),
        "eval_every": int(config.get("eval_every") or 0),
        "eval_episodes": int(config.get("eval_episodes") or 0),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started_ts)),
        "started_ts": started_ts,
        "elapsed": 0.0,
        "latest_update": updates[-1] if updates else None,
        "evaluations": evaluations[-10:],
        "summary": None,
        "live_file": _relative(live_path),
        "out_dir": _relative(job_dir),
        "error": None,
        "last_log": "",
        "pid": None,
        "adopted": True,
    }


def adoptable_job_dirs() -> list[Path]:
    """Run directories that look in-flight (live events, no final summary)."""

    found: list[Path] = []
    try:
        candidates = sorted(RUNS.glob("rwpp-ppo-*"))
    except OSError:
        return found
    for job_dir in candidates:
        if not job_dir.is_dir():
            continue
        if (job_dir / "rwpp_summary.json").is_file():
            continue
        live_path = job_dir / "live.jsonl"
        if not live_path.is_file():
            continue
        try:
            age = time.time() - live_path.stat().st_mtime
        except OSError:
            continue
        if age > STALE_SECONDS:
            continue
        found.append(job_dir)
    return found


def load_persisted_jobs() -> int:
    """Restore ``JOBS`` entries written by ``paths._update`` before the restart.

    A persisted record claiming ``running`` must be re-validated, otherwise every
    restart would resurrect jobs that finished days ago and they would sit in the
    dashboard forever. Validation is artifact-based, never a guess:

    * PPO: ``rwpp_summary.json`` -> ``done``; ``live.jsonl`` fresh or its pid
      alive -> keep running; otherwise ``stale``.
    * infer/train: no artifacts to consult, so a ``running`` record is left alone
      only when it is younger than ``STALE_SECONDS``.
    """

    restored = 0
    if not JOB_STATE_DIR.is_dir():
        return restored
    pids = _trainer_pids()
    for path in sorted(JOB_STATE_DIR.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(record, dict):
            continue
        job_id = record.get("id") or path.stem
        record["id"] = job_id
        if record.get("status") == "running":
            _revalidate(record, pids)
        with JOBS_LOCK:
            if job_id in JOBS:
                continue
            JOBS[job_id] = record
        restored += 1
    return restored


def _revalidate(record: dict, pids: dict[str, int]) -> None:
    """Downgrade a persisted ``running`` record that no longer has live evidence."""

    out_dir = record.get("out_dir")
    if record.get("kind") == "ppo" and out_dir:
        job_dir = Path(REPO_ROOT / str(out_dir))
        live_path = job_dir / "live.jsonl"
        if (job_dir / "rwpp_summary.json").is_file():
            record["status"] = "done"
            record["finished_at"] = record.get("finished_at") or time.strftime("%Y-%m-%d %H:%M:%S")
            return
        pid = pids.get(str(job_dir.resolve()))
        fresh = False
        try:
            fresh = (time.time() - live_path.stat().st_mtime) <= STALE_SECONDS
        except OSError:
            fresh = False
        if pid:
            record["pid"] = pid
            record["adopted"] = True
            return
        if fresh:
            record["adopted"] = True
            return
        record["status"] = "stale"
        record["error"] = record.get("error") or "服务重启后未发现仍在运行的训练进程"
        record["finished_at"] = record.get("finished_at") or time.strftime("%Y-%m-%d %H:%M:%S")
        return
    started_ts = float(record.get("started_ts") or 0.0)
    if started_ts and (time.time() - started_ts) <= STALE_SECONDS:
        return
    record["status"] = "stale"
    record["error"] = record.get("error") or "服务重启，任务已中断"
    record["finished_at"] = record.get("finished_at") or time.strftime("%Y-%m-%d %H:%M:%S")


def _monitor_adopted(job_id: str) -> None:
    """Keep one adopted job's progress fresh until its artifacts say it ended."""

    with JOBS_LOCK:
        job = JOBS.get(job_id)
        live_file = (job or {}).get("live_file")
        out_dir = (job or {}).get("out_dir")
    if not live_file:
        return
    live_path = (REPO_ROOT / live_file).resolve()
    if REPO_ROOT.resolve() not in live_path.parents:
        return
    job_dir = live_path.parent
    summary_path = job_dir / "rwpp_summary.json"
    log_path = job_dir / "train.log"
    started = float((job or {}).get("started_ts") or time.time())

    while job_id not in state.PPO_STOP:
        time.sleep(POLL_SECONDS)
        events = _read_events(live_path)
        updates = [event["payload"] for event in events if event.get("kind") == "update"]
        evaluations = [event["payload"] for event in events if event.get("kind") == "eval"]
        try:
            age = time.time() - live_path.stat().st_mtime
        except OSError:
            age = STALE_SECONDS + 1.0

        status = "running"
        error = None
        summary = None
        if summary_path.is_file():
            status = "done"
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                summary = None
        else:
            with JOBS_LOCK:
                pid = (JOBS.get(job_id) or {}).get("pid")
            if pid and not _pid_alive(int(pid)):
                # The trainer is gone and never wrote a summary: that is a failure.
                status = "failed"
                error = "adopted trainer process exited without rwpp_summary.json"
            elif age > STALE_SECONDS:
                status = "stale"
                error = (
                    f"no new events for {int(age)}s; the trainer may be wedged or was "
                    "killed - check runs/_rwpp_ppo/game-*.log"
                )

        log_tail = ""
        try:
            log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-1500:]
        except OSError:
            pass
        if status == "running" and "Traceback" in log_tail:
            status = "failed"
            lines = [line for line in log_tail.strip().splitlines() if line.strip()]
            error = lines[-1] if lines else "trainer traceback"

        fields = {
            "updates_done": len(updates),
            "latest_update": updates[-1] if updates else None,
            "evaluations": evaluations[-10:],
            "elapsed": round(time.time() - started, 1),
            "last_log": log_tail,
        }
        if status == "running":
            # Go through ``_update`` so the progress is persisted too: otherwise a
            # second restart would lose everything adopted since the first one.
            _update(job_id, **fields)
            continue

        fields.update(
            status=status,
            error=error,
            summary=summary,
            finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        _update(job_id, **fields)
        return


def recover_jobs() -> int:
    """Restore persisted jobs, then adopt any in-flight run found on disk.

    Called once from ``server.main()`` before the HTTP server starts listening,
    so the very first ``/jobs`` poll already shows the adopted run.
    """

    restored = load_persisted_jobs()

    known = set()
    with JOBS_LOCK:
        for job in JOBS.values():
            if job.get("out_dir"):
                known.add(str(job["out_dir"]))
        running_out = {
            str(job.get("out_dir"))
            for job in JOBS.values()
            if job.get("kind") == "ppo" and job.get("status") == "running"
        }

    pids, cmdlines = _scan_trainers()
    for job_dir in adoptable_job_dirs():
        out_dir = _relative(job_dir)
        if out_dir in known or out_dir in running_out:
            continue
        resolved = str(job_dir.resolve())
        record = _job_from_live(job_dir, cmdlines.get(resolved))
        if not record:
            continue
        record["pid"] = pids.get(resolved)
        with JOBS_LOCK:
            if record["id"] in JOBS:
                continue
            JOBS[record["id"]] = record
        threading.Thread(
            target=_monitor_adopted, args=(record["id"],), daemon=True, name=f"adopt-{record['id']}"
        ).start()
        restored += 1
    return restored
