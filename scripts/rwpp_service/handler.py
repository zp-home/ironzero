# SPDX-License-Identifier: Apache-2.0
"""手写 HTTP 路由层（``Handler(BaseHTTPRequestHandler)``）。

路由表（前缀与响应字段逐字节保持与拆分前一致）：

===========================  ======  =========================================
路径                         方法    说明
===========================  ======  =========================================
``/health``                  GET     服务健康 + 任务计数 + 空闲端口数
``/jobs``                    GET     最近 50 个任务（``{"jobs": [...]}``）
``/jobs/{id}``               GET     单任务记录（running 时刷新 elapsed）
``/ppo/{id}/events``         GET     PPO 实时事件增量（``?after=<seq>``）
``/infer``, ``/record``      POST    单局实机推理
``/train``                   POST    多局并行采集
``/ppo``                     POST    实机 PPO 训练
``/jobs/{id}/stop``          POST    停止单任务
``/jobs/stop_all``           POST    停止全部（别名 ``/jobs/stop-all``）
``/stop_all``                POST    停止全部（别名）
===========================  ======  =========================================

未知路径一律 404 ``{"error": "not found"}``。
"""

from __future__ import annotations

import json
import time

from http.server import BaseHTTPRequestHandler

from . import state
from .launch import StartMixin
from .paths import REPO_ROOT
from .ppo import _read_live_events
from .proc import stop_all_jobs, stop_job
from .state import JOBS, JOBS_LOCK

__all__ = ["Handler"]


class Handler(StartMixin, BaseHTTPRequestHandler):
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
                    "max_parallel": state.MAX_PARALLEL,
                    "free_slots": state.PORT_SLOTS.qsize(),
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
