# SPDX-License-Identifier: Apache-2.0
"""Host-side "real game" service for the IronZero platform.

The platform backend runs inside Docker without Java, so the real Rusted
Warfare 1.15 runtime lives on the Windows host behind this small HTTP service.
The platform calls it through ``host.docker.internal`` to record a real game
episode (RWPP headless + IronZero bridge extension + scripted orders); the
recording lands under ``runs/platform/real-rwpp-*/replays`` where the dashboard
replay library picks it up.

Run:

    python scripts/rwpp_real_service.py --port 8300
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

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
RUN_LOCK = threading.Lock()


def run_job(job_id: str, ticks: int, policy: str, map_name: str | None) -> None:
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
    started = time.time()
    try:
        result = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )
        replay_path = None
        for line in result.stdout.splitlines():
            if line.startswith("replay written:"):
                replay_path = line.split("replay written:", 1)[1].strip().split(" (")[0]
        status = "done" if result.returncode == 0 and replay_path else "failed"
        with JOBS_LOCK:
            JOBS[job_id].update(
                {
                    "status": status,
                    "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "seconds": round(time.time() - started, 1),
                    "replay_path": replay_path,
                    "log": (result.stdout or "")[-2000:] + (result.stderr or "")[-2000:],
                }
            )
    except Exception as exc:  # noqa: BLE001
        with JOBS_LOCK:
            JOBS[job_id].update(
                {"status": "failed", "error": str(exc), "seconds": round(time.time() - started, 1)}
            )


class Handler(BaseHTTPRequestHandler):
    server_version = "IronZeroRWPP/0.1"

    def log_message(self, fmt: str, *args) -> None:  # quieter logs
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
            self._send({"ok": True, "source": "host-rwpp", "jobs": len(JOBS)})
            return
        if self.path.startswith("/jobs/"):
            job_id = self.path.rsplit("/", 1)[-1]
            with JOBS_LOCK:
                job = JOBS.get(job_id)
            if not job:
                self._send({"error": "job not found"}, 404)
                return
            self._send(job)
            return
        self._send({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path.startswith("/record"):
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                self._send({"error": "invalid json"}, 400)
                return
            ticks = int(payload.get("ticks", 2400))
            policy = str(payload.get("policy", "scripted"))
            map_name = payload.get("map_name")
            if ticks < 120 or ticks > 20000:
                self._send({"error": "ticks out of range"}, 400)
                return
            if not RUN_LOCK.acquire(blocking=False):
                self._send({"error": "another real-game recording is running"}, 409)
                return
            job_id = f"rwpp-{uuid.uuid4().hex[:8]}"
            with JOBS_LOCK:
                JOBS[job_id] = {
                    "id": job_id,
                    "status": "running",
                    "ticks": ticks,
                    "policy": policy,
                    "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            def runner() -> None:
                try:
                    run_job(job_id, ticks, policy, map_name)
                finally:
                    RUN_LOCK.release()
            threading.Thread(target=runner, daemon=True).start()
            self._send({"job_id": job_id, "status": "running"})
            return
        self._send({"error": "not found"}, 404)


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
