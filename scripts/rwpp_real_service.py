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
RUNS = REPO_ROOT / "runs" / "platform"
RUNS.mkdir(parents=True, exist_ok=True)

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
PORT_SLOTS: "queue.Queue[int]" = queue.Queue()
MAX_PARALLEL = 4


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
) -> tuple[str | None, int, float]:
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
        "300",
    ]
    if map_name:
        command.extend(["--map-name", map_name])
    if stream_path is not None:
        command.extend(["--stream", str(stream_path)])
    started = time.time()
    result = subprocess.run(
        command, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=360
    )
    seconds = time.time() - started
    replay_path = None
    frame_count = 0
    for line in result.stdout.splitlines():
        if line.startswith("replay written:"):
            payload = line.split("replay written:", 1)[1].strip()
            replay_path = payload.split(" (")[0]
            if "(" in payload:
                try:
                    frame_count = int(payload.split("(")[1].split(" ")[0])
                except (IndexError, ValueError):
                    frame_count = 0
    log_tail = ((result.stdout or "") + (result.stderr or ""))[-1200:]
    _update(job_id, last_log=log_tail)
    return replay_path, frame_count, seconds


def run_infer(job_id: str, ticks: int, policy: str, map_name: str | None) -> None:
    port = PORT_SLOTS.get()
    # Publish the bridge port immediately so the dashboard can show it while the
    # episode is still running (not only after it completes).
    _update(job_id, port=port)
    job_dir = RUNS / f"rwpp-live-{job_id}"
    job_dir.mkdir(parents=True, exist_ok=True)
    frames_path = job_dir / "frames.jsonl"
    started = time.time()
    try:
        replay, frame_count, seconds = _run_recorder(
            job_id,
            ticks=ticks,
            policy=policy,
            map_name=map_name,
            stream_path=frames_path,
            port=port,
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
                try:
                    episode = work_queue.get_nowait()
                except queue.Empty:
                    return
                replay, frame_count, seconds = _run_recorder(
                    job_id,
                    ticks=ticks,
                    policy=policy,
                    map_name=map_name,
                    stream_path=None,
                    port=port,
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


class Handler(BaseHTTPRequestHandler):
    server_version = "IronZeroRWPP/0.3"

    def log_message(self, fmt: str, *args) -> None:
        print(f"[rwpp-service] {fmt % args}", flush=True)

    def _send(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/health"):
            with JOBS_LOCK:
                running = sum(1 for job in JOBS.values() if job["status"] == "running")
            self._send(
                {
                    "ok": True,
                    "source": "host-rwpp",
                    "jobs": len(JOBS),
                    "running": running,
                    "max_parallel": MAX_PARALLEL,
                    "free_slots": PORT_SLOTS.qsize(),
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
        self._send({"error": "not found"}, 404)

    def _start(self, kind: str, payload: dict) -> None:
        ticks = int(payload.get("ticks", 2400))
        policy = str(payload.get("policy", "scripted"))
        map_name = payload.get("map_name")
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
            }

        def runner() -> None:
            if kind == "train":
                run_train(job_id, episodes, ticks, policy, map_name, concurrency)
            else:
                run_infer(job_id, ticks, policy, map_name)

        threading.Thread(target=runner, daemon=True).start()
        self._send({"job_id": job_id, "status": "running", "kind": kind})


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
