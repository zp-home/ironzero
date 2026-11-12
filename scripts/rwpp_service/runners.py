# SPDX-License-Identifier: Apache-2.0
"""``/infer`` 与 ``/train`` 的执行体：拼命令行、跑子进程、回填任务记录。

* ``_run_recorder``  用 ``scripts/record_rwpp_replay.py`` 跑一局（scripted/none）
* ``_run_player``    用 ``scripts/play_rwpp_policy.py`` 跑一局（model/ppo checkpoint）
* ``run_infer``      单局推理：占端口 → 选 runner → 回填 status/replay/frames
* ``run_train``      多局采集：worker 线程池，共享端口池与任务计数器

端口生命周期：``run_infer`` / ``run_train`` 的每个 worker 都从 ``PORT_SLOTS``
取一个端口，结束（含异常）后归还，保证 ``--max-parallel`` 个并发实例。
"""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
from pathlib import Path

from . import state
from .paths import PLAYER, RECORDER, RUNS, _append_replay, _relative, _update
from .proc import _episode_timeout, _run_managed_proc
from .state import JOBS, JOBS_LOCK, PORT_SLOTS, STOPPED_JOBS

__all__ = ["_run_player", "_run_recorder", "run_infer", "run_train"]


def _run_recorder(
    job_id: str,
    *,
    ticks: int,
    policy: str,
    map_name: str | None,
    stream_path: Path | None,
    port: int,
    map_path: str | None = None,
    map_width: float = 2600.0,
    map_height: float = 2600.0,
    frame_ms: int = 2,
) -> tuple[str | None, int, float, str | None]:
    command = [
        sys.executable,
        str(RECORDER),
        "--ticks",
        str(ticks),
        "--policy",
        policy,
        "--port",
        str(port),
        "--timeout",
        str(_episode_timeout(ticks, frame_ms)),
        "--frame-ms",
        str(max(0, int(frame_ms))),
        "--map-width",
        str(float(map_width)),
        "--map-height",
        str(float(map_height)),
    ]
    if map_path:
        command.extend(["--map", map_path])
    if map_name:
        command.extend(["--map-name", map_name])
    if stream_path is not None:
        command.extend(["--stream", str(stream_path)])
    started = time.time()
    code, stdout = _run_managed_proc(
        job_id, command, timeout=_episode_timeout(ticks, frame_ms) + 60.0
    )
    seconds = time.time() - started
    replay_path = None
    frame_count = 0
    episode_result: str | None = None
    for line in stdout.splitlines():
        if line.startswith("rwpp-result:"):
            # The recorder prints a machine-readable summary with the real
            # bridge result (victory/defeat/timeout); use it for job metrics.
            try:
                payload = json.loads(line.split("rwpp-result:", 1)[1].strip())
            except ValueError:
                payload = {}
            if payload.get("result"):
                episode_result = str(payload["result"])
            if payload.get("replay"):
                replay_path = str(payload["replay"])
            continue
        if line.startswith("replay written:"):
            payload = line.split("replay written:", 1)[1].strip()
            replay_path = payload.split(" (")[0]
            if "(" in payload:
                try:
                    frame_count = int(payload.split("(")[1].split(" ")[0])
                except (IndexError, ValueError):
                    frame_count = 0
    log_tail = stdout[-1200:]
    _update(job_id, last_log=log_tail)
    return replay_path, frame_count, seconds, episode_result


def _run_player(
    job_id: str,
    *,
    ticks: int,
    checkpoint: str,
    map_name: str | None,
    map_path: str | None,
    map_width: float,
    map_height: float,
    stream_path: Path,
    port: int,
    frame_ms: int,
) -> tuple[str | None, int, float, str | None]:
    command = [
        sys.executable,
        str(PLAYER),
        "--checkpoint",
        checkpoint,
        "--ticks",
        str(ticks),
        "--port",
        str(port),
        "--timeout",
        str(_episode_timeout(ticks, frame_ms)),
        "--stream",
        str(stream_path),
        "--frame-ms",
        str(max(0, int(frame_ms))),
        "--map-width",
        str(float(map_width)),
        "--map-height",
        str(float(map_height)),
    ]
    if map_path:
        command.extend(["--map", map_path])
    if map_name:
        command.extend(["--map-name", map_name])
    if "ironzero" in str(map_path or "").lower():
        command.extend(["--win-mode", "annihilation", "--unlock-ai"])
    started = time.time()
    code, stdout = _run_managed_proc(
        job_id, command, timeout=_episode_timeout(ticks, frame_ms) + 60.0
    )
    seconds = time.time() - started
    payload: dict = {}
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                payload = json.loads(line)  # last JSON line wins (the summary)
            except ValueError:
                continue
    log_tail = stdout[-1500:]
    _update(job_id, last_log=log_tail)
    if not payload.get("replay"):
        return None, 0, seconds, None
    episode_result = str(payload["result"]) if payload.get("result") else None
    return str(payload["replay"]), int(payload.get("frames", 0)), seconds, episode_result


def run_infer(
    job_id: str,
    ticks: int,
    policy: str,
    map_name: str | None,
    map_path: str | None = None,
    map_width: float = 2600.0,
    map_height: float = 2600.0,
    checkpoint: str | None = None,
    frame_ms: int = 2,
) -> None:
    port = PORT_SLOTS.get()
    # Publish the bridge port immediately so the dashboard can show it while the
    # episode is still running (not only after it completes).
    _update(job_id, port=port)
    job_dir = RUNS / f"rwpp-live-{job_id}"
    job_dir.mkdir(parents=True, exist_ok=True)
    frames_path = job_dir / "frames.jsonl"
    started = time.time()
    try:
        if policy in ("model", "ppo") and checkpoint:
            replay, frame_count, seconds, episode_result = _run_player(
                job_id,
                ticks=ticks,
                checkpoint=checkpoint,
                map_name=map_name,
                map_path=map_path,
                map_width=map_width,
                map_height=map_height,
                stream_path=frames_path,
                port=port,
                frame_ms=frame_ms,
            )
        else:
            replay, frame_count, seconds, episode_result = _run_recorder(
                job_id,
                ticks=ticks,
                policy="none" if policy == "model" else policy,
                map_name=map_name,
                stream_path=frames_path,
                port=port,
                map_path=map_path,
                map_width=map_width,
                map_height=map_height,
                frame_ms=frame_ms,
            )
        _update(
            job_id,
            status="done" if replay else "failed",
            finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            seconds=round(time.time() - started, 1),
            replay_path=_relative(Path(replay)) if replay else None,
            frames_file=_relative(frames_path),
            frames=frame_count,
            port=port,
            # Real bridge outcome (victory/defeat/timeout) for the dashboard.
            result=episode_result,
        )
    except Exception as exc:  # noqa: BLE001
        _update(job_id, status="failed", error=str(exc), seconds=round(time.time() - started, 1))
    finally:
        PORT_SLOTS.put(port)


def run_train(
    job_id: str,
    episodes: int,
    ticks: int,
    policy: str,
    map_name: str | None,
    concurrency: int,
    map_path: str | None = None,
    map_width: float = 2600.0,
    map_height: float = 2600.0,
) -> None:
    started = time.time()
    workers = max(1, min(concurrency, episodes, state.MAX_PARALLEL))
    work_queue: "queue.Queue[int]" = queue.Queue()
    for episode in range(episodes):
        work_queue.put(episode)

    def worker() -> None:
        port = PORT_SLOTS.get()
        try:
            while True:
                if job_id in STOPPED_JOBS:
                    return
                try:
                    episode = work_queue.get_nowait()
                except queue.Empty:
                    return
                if job_id in STOPPED_JOBS:
                    return
                replay, frame_count, seconds, episode_result = _run_recorder(
                    job_id,
                    ticks=ticks,
                    policy=policy,
                    map_name=map_name,
                    stream_path=None,
                    port=port,
                    map_path=map_path,
                    map_width=map_width,
                    map_height=map_height,
                )
                if replay:
                    _append_replay(job_id, _relative(Path(replay)))
                with JOBS_LOCK:
                    job = JOBS[job_id]
                    job["episodes_done"] += 1
                    job["episodes_metrics"].append(
                        {
                            "episode": episode,
                            "seconds": round(seconds, 1),
                            "ticks": ticks,
                            "frames": frame_count,
                            # Real bridge outcome per episode (victory/defeat/timeout).
                            "result": episode_result,
                            "replay": _relative(Path(replay)) if replay else None,
                        }
                    )
                    job["frames"] = job.get("frames", 0) + frame_count
                    job["ticks_total"] = job.get("ticks_total", 0) + ticks
                    job["elapsed"] = round(time.time() - started, 1)
        finally:
            PORT_SLOTS.put(port)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(workers)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        with JOBS_LOCK:
            total = len(JOBS[job_id]["replays"])
        _update(
            job_id,
            status="done" if total else "failed",
            finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            seconds=round(time.time() - started, 1),
        )
    except Exception as exc:  # noqa: BLE001
        _update(job_id, status="failed", error=str(exc), seconds=round(time.time() - started, 1))
