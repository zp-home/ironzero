# SPDX-License-Identifier: Apache-2.0
"""Policy checkpoint discovery."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from ..jobs import _find_checkpoints

router = APIRouter()


@router.get("/api/checkpoints")
def list_checkpoints() -> list[dict[str, Any]]:
    """List available policy checkpoints under the configured roots.

    Delegates to ``_find_checkpoints`` so the search paths stay portable
    (``IRONZERO_CHECKPOINT_ROOTS`` / repo-relative) instead of the previous
    hard-coded ``/app/runs`` container path.
    """

    return _find_checkpoints()
