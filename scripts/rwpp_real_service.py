# SPDX-License-Identifier: Apache-2.0
"""Host-side "real game" service for the IronZero platform.

The platform backend runs inside Docker without Java, so the real Rusted
Warfare 1.15 runtime lives on the Windows host behind this small HTTP service.

Endpoints:

* ``GET  /health``
* ``POST /infer  {ticks, policy, map_name}``  one real-game episode, live frames
* ``POST /train  {episodes, ticks, policy, map_name, concurrency}``  parallel data collection
* ``GET  /jobs/{id}`` job status, progress, per-episode metrics, frames/replay paths

Parallelism: each concurrent game instance is a separate RWPP process with its
own bridge port from the port pool; the game simulation itself is single-threaded.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RECORDER = REPO_ROOT / "scripts" / "record_rwpp_replay.py"
PLAYER = REPO_ROOT / "scripts" / "play_rwpp_policy.py"
RUNS = REPO_ROOT / "runs" / "platform"
RUNS.mkdir(parents=True, exist_ok=True)

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
PORT_SLOTS: "queue.Queue[int]" = queue.Queue()
MAX_PARALLEL = 4

#: Wall-clock cost of one simulated tick *beyond* the pacing sleep. Measured on
#: the challenge maps, the engine + bridge + client pipeline adds ~6 ms per tick.
TICK_OVERHEAD_MS = 6.0
#: Extra wall clock for engine startup, map load and the final replay write.
EPISODE_TIMEOUT_MARGIN_S = 90.0


ACTIVE_PROCESSES: dict[str, list[subprocess.Popen]] = {}
STOPPED_JOBS: set[str] = set()


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
        for p in procs:
            try:
                p.kill()
            except OSError:
                pass
        ACTIVE_PROCESSES[job_id] = []
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


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _update(job_id: str, **fields) -> None:
    with JOBS_LOCK:
        JOBS[job_id].update(fields)


def _append_replay(job_id: str, replay: str) -> None:
    with JOBS_LOCK:
        JOBS[job_id]["replays"].append(replay)


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
    workers = max(1, min(concurrency, episodes, MAX_PARALLEL))
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


# ----------------------------------------------------------------------
# PPO jobs: strategy-in-the-loop training on real games
# ----------------------------------------------------------------------
PPO_PROCESSES: dict[str, subprocess.Popen] = {}
PPO_STOP: set[str] = set()


def _read_live_events(path: Path) -> list[dict]:
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


def run_ppo(
    job_id: str,
    updates: int,
    envs: int,
    rollout_steps: int,
    episode_ticks: int,
    device: str,
    map_name: str,
    map_path: str,
    map_width: float,
    map_height: float,
    frame_ms: int = 2,
    vision_radius: float = 260.0,
    eval_every: int = 0,
    eval_episodes: int = 1,
    use_llm: bool = False,
    units_per_team: int | None = None,
    max_entities: int | None = None,
    target_slots: int | None = None,
    terrain: str | None = None,
    win_mode: str = "",
    unlock_ai: bool = False,
    init_checkpoint: str | None = None,
) -> None:
    job_dir = RUNS / f"rwpp-ppo-{job_id}"
    job_dir.mkdir(parents=True, exist_ok=True)
    live_path = job_dir / "live.jsonl"
    log_path = job_dir / "train.log"
    summary_path = job_dir / "rwpp_summary.json"
    command = [
        sys.executable,
        "-m",
        "ironzero.training.train_rwpp",
        "--updates",
        str(updates),
        "--envs",
        str(envs),
        "--rollout-steps",
        str(rollout_steps),
        "--episode-ticks",
        str(episode_ticks),
        "--device",
        device,
        "--map",
        map_path,
        "--map-name",
        map_name,
        "--map-width",
        str(map_width),
        "--map-height",
        str(map_height),
        "--frame-ms",
        str(max(0, int(frame_ms))),
        "--vision-radius",
        str(float(vision_radius)),
        "--out",
        str(job_dir),
        "--live",
        str(live_path),
        "--eval-every",
        str(int(eval_every) if int(eval_every) > 0 else int(updates)),
        "--eval-episodes",
        str(max(1, int(eval_episodes))),
    ]
    if use_llm:
        command.append("--use-llm")
    # Arena scenarios (e.g. the 25-vs-25 IronZero25 map) need the wider
    # observation/action contract, the terrain channel and the bridge's
    # annihilation win condition; legacy maps keep the old 8-unit defaults.
    if units_per_team:
        command.extend(["--units-per-team", str(int(units_per_team))])
    if max_entities:
        command.extend(["--max-entities", str(int(max_entities))])
    if target_slots:
        command.extend(["--target-slots", str(int(target_slots))])
    if terrain:
        command.extend(["--terrain", str(terrain)])
    if win_mode:
        command.extend(["--win-mode", str(win_mode)])
    if unlock_ai:
        command.append("--unlock-ai")
    if init_checkpoint:
        command.extend(["--init-checkpoint", str(init_checkpoint)])
    started = time.time()
    handle = log_path.open("w", encoding="utf-8", errors="replace")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    environment["PYTHONUNBUFFERED"] = "1"
    try:
        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        PPO_PROCESSES[job_id] = process
        _update(job_id, pid=process.pid, live_file=_relative(live_path), out_dir=_relative(job_dir))
        while process.poll() is None:
            time.sleep(2.0)
            events = _read_live_events(live_path)
            updates_done = sum(1 for event in events if event.get("kind") == "update")
            evaluations = [event["payload"] for event in events if event.get("kind") == "eval"]
            latest_update = next(
                (event["payload"] for event in reversed(events) if event.get("kind") == "update"), None
            )
            _update(
                job_id,
                updates_done=updates_done,
                latest_update=latest_update,
                evaluations=evaluations[-10:],
                elapsed=round(time.time() - started, 1),
                events=len(events),
            )
        return_code = process.wait()
        events = _read_live_events(live_path)
        updates_done = sum(1 for event in events if event.get("kind") == "update")
        summary = None
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                summary = None
        log_tail = ""
        try:
            log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-1500:]
        except OSError:
            pass
        stopped = job_id in PPO_STOP
        status = "stopped" if stopped else ("done" if return_code == 0 else "failed")
        _update(
            job_id,
            status=status,
            finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            seconds=round(time.time() - started, 1),
            elapsed=round(time.time() - started, 1),
            updates_done=updates_done,
            latest_update=next(
                (event["payload"] for event in reversed(events) if event.get("kind") == "update"), None
            ),
            evaluations=[event["payload"] for event in events if event.get("kind") == "eval"][-10:],
            summary=summary,
            last_log=log_tail,
            error=None if (return_code == 0 or stopped) else f"trainer exited with code {return_code}",
        )
    except Exception as exc:  # noqa: BLE001
        _update(job_id, status="failed", error=str(exc), elapsed=round(time.time() - started, 1))
    finally:
        PPO_STOP.discard(job_id)
        handle.close()
        PPO_PROCESSES.pop(job_id, None)


class Handler(BaseHTTPRequestHandler):
    server_version = "IronZeroRWPP/0.3"

    def log_message(self, fmt: str, *args) -> None:
        print(f"[rwpp-service] {fmt % args}", flush=True)

    def _send(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/health"):
            with JOBS_LOCK:
                running = sum(1 for job in JOBS.values() if job["status"] == "running")
                ppo_running = sum(
                    1
                    for job in JOBS.values()
                    if job.get("kind") == "ppo" and job.get("status") == "running"
                )
            self._send(
                {
                    "ok": True,
                    "source": "host-rwpp",
                    "jobs": len(JOBS),
                    "running": running,
                    "ppo_running": ppo_running,
                    "max_parallel": MAX_PARALLEL,
                    "free_slots": PORT_SLOTS.qsize(),
                }
            )
            return
        if self.path.startswith("/ppo/") and "/events" in self.path:
            job_id = self.path.split("?")[0].split("/ppo/")[1].split("/")[0]
            query = self.path.split("?", 1)[1] if "?" in self.path else ""
            after = 0
            for part in query.split("&"):
                if part.startswith("after="):
                    try:
                        after = int(part.split("=", 1)[1])
                    except ValueError:
                        after = 0
            with JOBS_LOCK:
                job = dict(JOBS.get(job_id)) if job_id in JOBS else None
            if not job:
                self._send({"error": "job not found"}, 404)
                return
            live_file = job.get("live_file")
            events: list[dict] = []
            if live_file:
                path = (REPO_ROOT / live_file).resolve()
                if REPO_ROOT.resolve() in path.parents:
                    events = _read_live_events(path)
            pending = [event for event in events if int(event.get("seq", 0)) > after]
            self._send(
                {
                    "next": (int(pending[-1].get("seq", after)) if pending else after),
                    "events": pending,
                    "status": job.get("status"),
                    "job": {
                        key: job.get(key)
                        for key in (
                            "id",
                            "kind",
                            "status",
                            "updates_total",
                            "updates_done",
                            "envs",
                            "rollout_steps",
                            "episode_ticks",
                            "device",
                            "frame_ms",
                            "vision_radius",
                            "map_name",
                            "elapsed",
                            "latest_update",
                            "evaluations",
                            "summary",
                            "error",
                        )
                    },
                }
            )
            return
        if self.path.startswith("/jobs/"):
            job_id = self.path.split("?")[0].rsplit("/", 1)[-1]
            with JOBS_LOCK:
                job = dict(JOBS.get(job_id)) if job_id in JOBS else None
                if job is not None and job.get("status") == "running":
                    started_ts = float(job.get("started_ts") or 0.0)
                    if started_ts:
                        job["elapsed"] = round(time.time() - started_ts, 1)
            if not job:
                self._send({"error": "job not found"}, 404)
                return
            self._send(job)
            return
        if self.path.startswith("/jobs"):
            with JOBS_LOCK:
                jobs = [dict(job) for job in JOBS.values()]
            for job in jobs:
                if job.get("status") == "running":
                    started_ts = float(job.get("started_ts") or 0.0)
                    if started_ts:
                        job["elapsed"] = round(time.time() - started_ts, 1)
            self._send({"jobs": jobs[-50:]})
            return
        self._send({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._send({"error": "invalid json"}, 400)
            return
        if self.path.startswith("/infer") or self.path.startswith("/record"):
            self._start(kind="infer", payload=payload)
            return
        if self.path.startswith("/train"):
            self._start(kind="train", payload=payload)
            return
        if self.path.startswith("/ppo"):
            self._start_ppo(payload)
            return
        if self.path.startswith("/jobs/") and self.path.rstrip("/").endswith("/stop"):
            job_id = self.path.split("?")[0].rstrip("/").rsplit("/", 2)[-2]
            stopped = stop_job(job_id)
            if not stopped:
                self._send({"error": f"job not found: {job_id}"}, 404)
                return
            self._send({"ok": True, "job_id": job_id, "stopped": True})
            return
        if self.path.rstrip("/") in ("/jobs/stop_all", "/jobs/stop-all", "/stop_all"):
            count = stop_all_jobs()
            self._send({"ok": True, "stopped_count": count})
            return
        self._send({"error": "not found"}, 404)

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
        # Arena preset: configurable tank counts (1..25 per side)
        units_per_team = payload.get("units_per_team")
        arena = (
            bool(payload.get("arena"))
            or "ironzero" in map_path.lower()
            or "ironzero" in map_name.lower()
        )
        max_entities = payload.get("max_entities")
        target_slots = payload.get("target_slots")
        terrain = str(payload.get("terrain") or "")
        win_mode = str(payload.get("win_mode") or "")
        unlock_ai = bool(payload.get("unlock_ai", False))
        if arena:
            if units_per_team is not None:
                units_per_team = max(1, min(25, int(units_per_team)))
            else:
                digits = "".join(filter(str.isdigit, map_name))
                units_per_team = int(digits) if digits and 1 <= int(digits) <= 25 else 25
            max_entities = int(max_entities or units_per_team)
            target_slots = int(target_slots or (max_entities + 1))
            try:
                import sys
                scripts_dir = str(REPO_ROOT / "scripts")
                if scripts_dir not in sys.path:
                    sys.path.insert(0, scripts_dir)
                from make_ironzero_map import ensure_arena_map
                _, rel_path, arena_map_name = ensure_arena_map(units_per_team)
                map_path = rel_path
                map_name = arena_map_name
            except Exception:
                pass
            terrain = terrain or str(REPO_ROOT / "configs" / "maps" / f"{map_name}.terrain.json")
            if not Path(terrain).is_file():
                terrain = str(REPO_ROOT / "configs" / "maps" / "IronZero25.terrain.json")
            win_mode = win_mode or "annihilation"
            unlock_ai = True
        else:
            if units_per_team is not None:
                units_per_team = max(1, min(25, int(units_per_team)))
            if not terrain:
                candidate_terrain = REPO_ROOT / "configs" / "maps" / f"{Path(map_name).stem}.terrain.json"
                if candidate_terrain.is_file():
                    terrain = str(candidate_terrain)
                else:
                    terrain = ""

        init_checkpoint = payload.get("init_checkpoint")
        init_mode = payload.get("init_mode")
        if (init_mode == "warmstart" or not init_checkpoint) and init_mode != "scratch":
            # 根据当前坦克数量优先寻找最匹配的预热权重
            best_bc = REPO_ROOT / "runs" / f"bc_ironzero{units_per_team}" / "checkpoint.pt"
            if not best_bc.is_file():
                best_bc = REPO_ROOT / "runs" / "bc_ironzero25" / "checkpoint.pt"
            if best_bc.is_file():
                init_checkpoint = str(best_bc)
        if init_checkpoint:
            candidate = Path(str(init_checkpoint))
            if not candidate.is_absolute():
                candidate = REPO_ROOT / candidate
            if candidate.is_file():
                init_checkpoint = str(candidate.resolve())
            else:
                init_checkpoint = None

        if not 0 <= frame_ms <= 16:
            self._send({"error": "frame_ms must be 0..16"}, 400)
            return
        if not 0.0 <= vision_radius <= 2000.0:
            self._send({"error": "vision_radius must be 0..2000"}, 400)
            return
        if not 1 <= updates <= 500:
            self._send({"error": "updates must be 1..500"}, 400)
            return
        if not 1 <= envs <= 8:
            self._send({"error": "envs must be 1..8"}, 400)
            return
        if not 1 <= rollout_steps <= 512:
            self._send({"error": "rollout_steps must be 1..512"}, 400)
            return
        if not 120 <= episode_ticks <= 20000:
            self._send({"error": "episode_ticks must be 120..20000"}, 400)
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
        if not 1 <= concurrency <= MAX_PARALLEL:
            self._send({"error": f"concurrency must be 1..{MAX_PARALLEL}"}, 400)
            return
        if not 120 <= ticks <= 20000:
            self._send({"error": "ticks out of range"}, 400)
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


def main(argv: list[str] | None = None) -> int:
    global MAX_PARALLEL
    parser = argparse.ArgumentParser(description="IronZero host-side real-game service")
    parser.add_argument("--port", type=int, default=8300)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--max-parallel", type=int, default=4, help="maximum concurrent game instances")
    args = parser.parse_args(argv)
    MAX_PARALLEL = max(1, args.max_parallel)
    for slot in range(MAX_PARALLEL):
        PORT_SLOTS.put(5130 + slot)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"[rwpp-service] listening on http://{args.host}:{args.port} max_parallel={MAX_PARALLEL}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[rwpp-service] stopping", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
