# SPDX-License-Identifier: Apache-2.0
"""Filesystem roots and path-safety helpers for the platform backend.

Everything here is derived from the environment at import time, mirroring the
historical ``web/backend/app.py`` module-level constants: the repository root,
the job/artifact root (``IRONZERO_RUNS``), the dashboard frontend directory and
the host-side real-game service URL.
"""

from __future__ import annotations

import datetime
import os
from pathlib import Path

from fastapi import HTTPException

APP_VERSION = "0.1.0"
BEIJING_TZ = datetime.timezone(datetime.timedelta(hours=8))
REPO_ROOT = Path(os.environ.get("IRONZERO_REPO_ROOT", Path(__file__).resolve().parents[3])).resolve()
RUNS_ROOT = Path(os.environ.get("IRONZERO_RUNS", REPO_ROOT / "runs" / "platform")).resolve()


def _resolve_checkpoint_roots() -> list[Path]:
    """Checkpoint search roots, overridable and portable.

    ``IRONZERO_CHECKPOINT_ROOTS`` accepts a ``os.pathsep``-separated list (``;``
    on Windows, ``:`` on POSIX) so Docker can point at ``/app/runs`` while a
    local checkout keeps scanning its own ``runs/`` tree. Only directories that
    actually exist are kept, which avoids walking non-existent container paths.
    """

    configured = os.environ.get("IRONZERO_CHECKPOINT_ROOTS", "").strip()
    if configured:
        candidates = [Path(part).expanduser() for part in configured.split(os.pathsep) if part.strip()]
    else:
        candidates = [REPO_ROOT / "runs", RUNS_ROOT]
    roots: list[Path] = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.is_dir() and resolved not in roots:
            roots.append(resolved)
    return roots


CHECKPOINT_ROOTS = _resolve_checkpoint_roots()
FRONTEND_DIR = REPO_ROOT / "web" / "frontend"
RWPP_SERVICE = os.environ.get("IRONZERO_RWPP_SERVICE", "http://127.0.0.1:8300")
RUNS_ROOT.mkdir(parents=True, exist_ok=True)


def _resolve_inside(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if root != candidate and root not in candidate.parents:
        raise HTTPException(status_code=400, detail="path escapes the runs directory")
    return candidate
