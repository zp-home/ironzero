# SPDX-License-Identifier: Apache-2.0
"""Recorded episode library: index, fetch one replay, delete one replay."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from ..paths import BEIJING_TZ, RUNS_ROOT, _resolve_inside

router = APIRouter()


def _replay_index(root: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not root.is_dir():
        return items
    for path in root.rglob("replays/*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        meta = payload.get("meta") or {}
        relative = path.relative_to(root).as_posix()
        map_meta = meta.get("map") or {}
        items.append(
            {
                "id": relative,
                "name": path.name,
                "checkpoint": meta.get("checkpoint"),
                "opponent": meta.get("opponent"),
                "seed": meta.get("seed"),
                "result": meta.get("result"),
                "return": meta.get("return"),
                "episodes": meta.get("episodes") or len(payload.get("episodes", [])),
                "frames": len(payload.get("frames", [])),
                "created_at": meta.get("created_at") or datetime.datetime.fromtimestamp(path.stat().st_mtime, BEIJING_TZ).strftime("%Y-%m-%d %H:%M:%S"),
                "mode": meta.get("mode"),
                "sim_mode": meta.get("sim_mode", "arena"),
                "map_id": map_meta.get("id"),
                "map_name": map_meta.get("name"),
            }
        )
    items.sort(key=lambda item: (item.get("created_at") or "", item["id"]), reverse=True)
    return items


@router.get("/api/replays")
def list_replays(
    limit: int = Query(default=0, ge=0, le=1000),
    offset: int = Query(default=0, ge=0),
) -> Any:
    items = _replay_index(RUNS_ROOT)
    if limit > 0:
        return {
            "total": len(items),
            "offset": offset,
            "limit": limit,
            "items": items[offset : offset + limit],
        }
    return items


@router.get("/api/replays/{replay_id:path}")
def get_replay(replay_id: str) -> dict[str, Any]:
    path = _resolve_inside(RUNS_ROOT, replay_id)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="replay not found")
    return json.loads(path.read_text(encoding="utf-8"))


@router.delete("/api/replays/{replay_id:path}")
def delete_replay(replay_id: str) -> dict[str, Any]:
    path = _resolve_inside(RUNS_ROOT, replay_id)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="replay not found")
    path.unlink()
    return {"ok": True}
