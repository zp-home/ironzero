# SPDX-License-Identifier: Apache-2.0
"""Host-side "real game" service for the IronZero platform.

The platform backend runs inside Docker without Java, so the real Rusted
Warfare 1.15 runtime lives on the Windows host behind this small HTTP service.

Endpoints:

* ``GET  /health``
* ``POST /infer  {ticks, policy, map_name}``  one real-game episode, live frames
* ``POST /train  {episodes, ticks, policy, map_name}``  sequential real-game data collection
* ``GET  /jobs/{id}`` job status, progress, frames/replay paths

The service writes into the repository ``runs/`` tree, which is mounted into
the platform container, so the dashboard can read frames and replays directly.
"""

from __future__ import annotations

import argparse
import json
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
RUN_LOCK = threading.Lock()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _update(job_id: str, **fields) -> None:
    with JOBS_LOCK:
        JOBS[job_id].update(fields)


def _run_recorder(
    job_id: str,
    *,
    ticks: int,
    policy: str,
    map_name: str | None,
    stream_path: Path | None,
) -> str | None:
    command = [
        sys.executable,
        str(RECORDER),
        "--ticks",
        str(ticks),
        "--policy",
        policy,
        "--timeout",
        "240",
    ]
    if map_name:
        command.extend(["--map-name", map_name])
    if stream_path is not None:
        command.extend(["--stream", str(stream_path)])
    result = subprocess.run(
        command, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=300
    )
    replay_path = None
    for line in result.stdout.splitlines():
        if line.startswith("replay written:"):
            replay_path = line.split("replay written:", 1)[1].strip().split(" (")[0]
    log_tail = ((result.stdout or "") + (result.stderr or ""))[-1500:]
    _update(job_id, last_log=log_tail)
    return replay_path


def run_infer(job_id: str, ticks: int, policy: str, map_name: str | None) -> None:
    started = time.time()
    job_dir = RUNS / f"rwpp-live-{job_id}"
    job_dir.mkdir(parents=True, exist_ok=True)
    frames_path = job_dir / "frames.jsonl"
    try:
        replay = _run_recorder(
            job_id, ticks=ticks, policy=policy, map_name=map_name, stream_path=frames_path
        )
        _update(
            job_id,
            status="done" if replay else "failed",
            finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            seconds=round(time.time() - started, 1),
            replay_path=_relative(Path(replay)) if replay else None,
            frames_file=_relative(frames_path),
        )
    except Exception as exc:  # noqa: BLE001
        _update(job_id, status="failed", error=str(exc), seconds=round(time.time() - started, 1))
    finally:
        RUN_LOCK.release()


def run_train(
    job_id: str, episodes: int, ticks: int, policy: str, map_name: str | None
) -> None:
    started = time.time()
    replays: list[str] = []
    try:
        for episode in range(episodes):
            _update(job_id, episodes_done=episode, current_episode=episode + 1)
            replay = _run_recorder(
                job_id, ticks=ticks, policy=policy, map_name=map_name, stream_path=None
            )
            if replay:
                replays.append(_relative(Path(replay)))
            _update(job_id, replays=replays, episodes_done=episode + 1)
        _update(
            job_id,
            status="done" if replays else "failed",
            finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            seconds=round(time.time() - started, 1),
        )
    except Exception as exc:  # noqa: BLE001
        _update(job_id, status="failed", error=str(exc), seconds=round(time.time() - started, 1))
    finally:
        RUN_LOCK.release()


class Handler(BaseHTTPRequestHandler):
    server_version = "IronZeroRWPP/0.2"

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
            self._send({"ok": True, "source": "host-rwpp", "jobs": len(JOBS), "running": running})
            return
        if self.path.startswith("/jobs/"):
            job_id = self.path.split("?")[0].rsplit("/", 1)[-1]
            with JOBS_LOCK:
                job = JOBS.get(job_id)
            if not job:
                self._send({"error": "job not found"}, 404)
                return
            self._send(job)
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
        if kind == "train":
            if not 1 <= episodes <= 50:
                self._send({"error": "episodes out of range"}, 400)
                return
        if not 120 <= ticks <= 20000:
            self._send({"error": "ticks out of range"}, 400)
            return
        if not RUN_LOCK.acquire(blocking=False):
            self._send({"error": "another real-game job is running"}, 409)
            return
        job_id = f"rwpp-{uuid.uuid4().hex[:8]}"
        with JOBS_LOCK:
            JOBS[job_id] = {
                "id": job_id,
                "kind": kind,
                "status": "running",
                "ticks": ticks,
                "policy": policy,
                "episodes_total": episodes if kind == "train" else 1,
                "episodes_done": 0,
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "replays": [],
                "replay_path": None,
                "frames_file": f"runs/platform/rwpp-live-{job_id}/frames.jsonl" if kind == "infer" else None,
            }

        def runner() -> None:
            try:
                if kind == "train":
                    run_train(job_id, episodes, ticks, policy, map_name)
                else:
                    run_infer(job_id, ticks, policy, map_name)
            finally:
                pass

        threading.Thread(target=runner, daemon=True).start()
        self._send({"job_id": job_id, "status": "running", "kind": kind})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="IronZero host-side real-game service")
    parser.add_argument("--port", type=int, default=8300)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[rwpp-service] listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[rwpp-service] stopping", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
