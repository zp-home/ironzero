# SPDX-License-Identifier: Apache-2.0
"""Real game (RWPP) proxy to the host-side service.

The platform container has no Java, so every real-game action is forwarded to
``IRONZERO_RWPP_SERVICE`` (default ``http://127.0.0.1:8300``). ``_rwpp_call``
keeps the upstream client errors (400/404/409/429) actionable and only maps
genuine transport problems to 502/503.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from ..jobs import _read_jsonl
from ..models import RwppPpoRequest, RwppRecordRequest
from ..paths import REPO_ROOT, RWPP_SERVICE

router = APIRouter()


def _rwpp_call(path: str, payload: dict[str, Any] | None = None, timeout: float = 10.0) -> dict:
    url = RWPP_SERVICE.rstrip("/") + path
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST" if data else "GET"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            detail = json.loads(raw).get("error") or raw
        except (ValueError, AttributeError):
            detail = raw
        # Preserve actionable client errors (busy slots, bad payload) instead of
        # masking them as a gateway failure.
        status = exc.code if exc.code in (400, 404, 409, 429) else 502
        raise HTTPException(status_code=status, detail=f"host rwpp service error: {detail}") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503,
            detail=f"host rwpp service unavailable ({RWPP_SERVICE}): {exc}",
        ) from exc


@router.get("/api/rwpp/status")
def rwpp_status() -> dict[str, Any]:
    health = _rwpp_call("/health", timeout=5.0)
    return {"service": RWPP_SERVICE, **health}


@router.post("/api/rwpp/record")
def rwpp_record(request: RwppRecordRequest) -> dict[str, Any]:
    return _rwpp_call("/record", request.model_dump(), timeout=20.0)


@router.post("/api/rwpp/infer")
def rwpp_infer(request: RwppRecordRequest) -> dict[str, Any]:
    payload = request.model_dump()
    payload.pop("episodes", None)
    payload.pop("concurrency", None)
    # Checkpoints are listed with container paths (/app/runs/...); the host-side
    # real-game service needs a path relative to the repository root.
    checkpoint = payload.get("checkpoint")
    if checkpoint and checkpoint.startswith("/app/"):
        payload["checkpoint"] = checkpoint[len("/app/"):]
    return _rwpp_call("/infer", payload, timeout=20.0)


@router.post("/api/rwpp/train")
def rwpp_train(request: RwppRecordRequest) -> dict[str, Any]:
    return _rwpp_call("/train", request.model_dump(), timeout=20.0)


@router.get("/api/rwpp/infer/{job_id}/frames")
def rwpp_infer_frames(job_id: str, after: int = Query(default=0, ge=0)) -> dict[str, Any]:
    job = _rwpp_call(f"/jobs/{job_id}", timeout=10.0)
    frames_file = job.get("frames_file")
    if not frames_file:
        raise HTTPException(status_code=404, detail="job has no frame stream")
    path = (REPO_ROOT / frames_file).resolve()
    if REPO_ROOT not in path.parents:
        raise HTTPException(status_code=400, detail="invalid frames path")
    stream = _read_jsonl(path)
    pending = stream[after:]
    return {
        "next": after + len(pending),
        "frames": pending,
        "status": job.get("status"),
        "job": {key: job.get(key) for key in ("id", "kind", "status", "episodes_done", "episodes_total", "seconds", "result", "replay_path", "frames_file")},
    }


@router.get("/api/rwpp/jobs")
def rwpp_jobs() -> dict[str, Any]:
    """Recent real-game jobs (running first) for the dashboard cards."""
    payload = _rwpp_call("/jobs", timeout=10.0)
    jobs = payload.get("jobs", [])
    jobs.sort(key=lambda job: (job.get("status") != "running", job.get("started_at") or ""), reverse=True)
    return {"jobs": jobs}


@router.get("/api/rwpp/jobs/{job_id}")
def rwpp_job(job_id: str) -> dict[str, Any]:
    return _rwpp_call(f"/jobs/{job_id}", timeout=10.0)


@router.post("/api/rwpp/ppo")
def rwpp_ppo(request: RwppPpoRequest) -> dict[str, Any]:
    """Start a real-game PPO training job on the host (one at a time)."""

    return _rwpp_call("/ppo", request.model_dump(), timeout=20.0)


@router.get("/api/rwpp/ppo/{job_id}/events")
def rwpp_ppo_events(job_id: str, after: int = Query(default=0, ge=0)) -> dict[str, Any]:
    return _rwpp_call(f"/ppo/{job_id}/events?after={after}", timeout=10.0)


@router.post("/api/rwpp/ppo/{job_id}/stop")
def rwpp_ppo_stop(job_id: str) -> dict[str, Any]:
    return _rwpp_call(f"/jobs/{job_id}/stop", {}, timeout=10.0)


@router.post("/api/rwpp/jobs/{job_id}/stop")
def rwpp_job_stop(job_id: str) -> dict[str, Any]:
    return _rwpp_call(f"/jobs/{job_id}/stop", {}, timeout=10.0)


@router.post("/api/rwpp/stop_all")
def rwpp_stop_all() -> dict[str, Any]:
    return _rwpp_call("/stop_all", {}, timeout=10.0)
