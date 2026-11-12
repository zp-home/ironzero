# SPDX-License-Identifier: Apache-2.0
"""Service health, the dashboard page and the favicon.

Two routers are exported on purpose: the favicon must be registered *after*
the ``/static`` mount so the historical route order is preserved. Paths are
disjoint, so this is purely about keeping the mounted app byte-identical.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..jobs import _jobs
from ..paths import APP_VERSION, FRONTEND_DIR, REPO_ROOT, RUNS_ROOT

router = APIRouter()
# Registered by server.create_app() after the /static mount.
favicon_router = APIRouter()


@router.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "version": APP_VERSION,
        "runs_root": str(RUNS_ROOT),
        "repo_root": str(REPO_ROOT),
        "jobs_running": sum(1 for job in _jobs.values() if job["process"].poll() is None),
    }


@router.get("/")
def index() -> FileResponse:
    page = FRONTEND_DIR / "index.html"
    if not page.is_file():
        raise HTTPException(status_code=503, detail="dashboard frontend is not installed")
    return FileResponse(page)


@favicon_router.get("/favicon.ico")
def favicon() -> FileResponse:
    icon = FRONTEND_DIR / "favicon.svg"
    if icon.is_file():
        return FileResponse(icon)
    raise HTTPException(status_code=404)
