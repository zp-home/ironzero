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

from ..jsonl_index import MAX_WINDOW_RECORDS, jsonl_index
from ..models import RwppPpoRequest, RwppRecordRequest
from ..paths import REPO_ROOT, RWPP_SERVICE

router = APIRouter()

#: Bound of ``fast_forward`` (engine ``gameSpeed``). 1 = off.
FAST_FORWARD_MIN = 1
FAST_FORWARD_MAX = 32


def _check_acceleration(payload: dict[str, Any]) -> None:
    """Refuse ``fast_forward`` outside 1..32 with the contract's wording.

    Deliberately a 400 and not a pydantic ``ge``/``le`` bound: a field bound
    answers 422 with pydantic's own sentence, so the dashboard would show one
    message while a direct curl to the host service (``scripts/rwpp_service``,
    which answers 400 ``fast_forward must be 1..32``) shows another. Same value,
    same status, same text on every entry point.
    """

    value = payload.get("fast_forward")
    if value is None:
        return
    try:
        speed = int(value)
    except (TypeError, ValueError):
        speed = None
    if speed is None or not FAST_FORWARD_MIN <= speed <= FAST_FORWARD_MAX:
        raise HTTPException(status_code=400, detail="fast_forward must be 1..32")


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
    payload = request.model_dump()
    _check_acceleration(payload)
    payload["checkpoint"] = _host_path(payload.get("checkpoint"))
    return _rwpp_call("/record", payload, timeout=20.0)


def _host_path(checkpoint: Any) -> Any:
    """Map a container path (``/app/runs/...``) back to the host's repo-relative form.

    The dashboard lists weights with the paths the *backend* sees, and inside the
    platform container that is ``/app/runs/...``. The real-game service runs on the
    host, where ``/app`` does not exist, so every forwarded checkpoint has to be
    translated. ``/infer`` has always done this; ``/ppo`` did not, and the host
    then treated a perfectly valid weight as "not found" (silently training from
    scratch instead of warm-starting).
    """

    if not checkpoint:
        return checkpoint
    text = str(checkpoint)
    if text.startswith("/app/"):
        return text[len("/app/"):]
    return text


@router.post("/api/rwpp/infer")
def rwpp_infer(request: RwppRecordRequest) -> dict[str, Any]:
    payload = request.model_dump()
    _check_acceleration(payload)
    payload.pop("episodes", None)
    payload.pop("concurrency", None)
    # Checkpoints are listed with container paths (/app/runs/...); the host-side
    # real-game service needs a path relative to the repository root.
    payload["checkpoint"] = _host_path(payload.get("checkpoint"))
    return _rwpp_call("/infer", payload, timeout=20.0)


@router.post("/api/rwpp/train")
def rwpp_train(request: RwppRecordRequest) -> dict[str, Any]:
    payload = request.model_dump()
    _check_acceleration(payload)
    payload["checkpoint"] = _host_path(payload.get("checkpoint"))
    return _rwpp_call("/train", payload, timeout=20.0)


@router.get("/api/rwpp/infer/{job_id}/frames")
def rwpp_infer_frames(
    job_id: str,
    after: int = Query(default=0, ge=0),
    limit: int | None = Query(
        default=None,
        ge=0,
        le=MAX_WINDOW_RECORDS,
        description=(
            "Records to return. Omit for the whole tail (unchanged behaviour); "
            "0 for a server-chosen bounded chunk; N to cap at N."
        ),
    ),
) -> dict[str, Any]:
    job = _rwpp_call(f"/jobs/{job_id}", timeout=10.0)
    frames_file = job.get("frames_file")
    if not frames_file:
        raise HTTPException(status_code=404, detail="job has no frame stream")
    path = (REPO_ROOT / frames_file).resolve()
    if REPO_ROOT not in path.parents:
        raise HTTPException(status_code=400, detail="invalid frames path")
    # Index-backed window read. A live episode's stream reaches 100+ MB and this
    # endpoint is polled every 700 ms *per concurrent instance*; parsing the whole
    # file each time (the previous behaviour) cost seconds of GIL inside the one
    # process that also serves /health and the static dashboard.
    frames, next_cursor = jsonl_index(path).read_window(after, limit)
    return {
        "next": next_cursor,
        "frames": frames,
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

    payload = request.model_dump()
    _check_acceleration(payload)
    payload["init_checkpoint"] = _host_path(payload.get("init_checkpoint"))
    # ``None`` means "let the host service decide" (warmstart lookup, engine
    # defaults). Forwarding the nulls keeps that intent explicit instead of
    # letting the host fall back to a stale default of its own.
    return _rwpp_call("/ppo", payload, timeout=20.0)


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
