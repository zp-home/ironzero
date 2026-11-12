# SPDX-License-Identifier: Apache-2.0
"""Real-game assets: map list, one parsed map, tileset/unit sprite files.

Reads only the local ``.game_cache`` copy through ``web/backend/maps.py``.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from ... import maps as gamemaps
from ... import unit_sprites as gamesprites

router = APIRouter()

_map_list_cache: dict[str, Any] = {"at": 0.0, "payload": None}


@router.get("/api/maps")
def api_maps(force: bool = False) -> dict[str, Any]:
    now = time.time()
    if (
        force
        or _map_list_cache["payload"] is None
        or now - float(_map_list_cache["at"]) > 30.0
    ):
        available = gamemaps.assets_available()
        payload = {
            "available": available,
            "assets_root": str(gamemaps.ASSETS_ROOT),
            "maps": gamemaps.list_maps() if available else [],
            # Legacy flat key -> relPath map. Kept for older frontends.
            "units": gamemaps.unit_sprites() if available else {},
            # Contract §1: full per-unit_type sprite manifest (docs/沙盘贴图契约.md).
            "unitSprites": (
                gamesprites.unit_sprites_manifest() if available else {}
            ),
        }
        _map_list_cache["payload"] = payload
        _map_list_cache["at"] = now
    return _map_list_cache["payload"]


@router.get("/api/maps/{map_id}")
def api_map(map_id: str) -> JSONResponse:
    try:
        payload = gamemaps.load_map(map_id)
    except gamemaps.MapError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return JSONResponse(payload)


@router.get("/api/assets/{asset_path:path}")
def api_asset(asset_path: str) -> FileResponse:
    try:
        path = gamemaps.resolve_asset(asset_path)
    except gamemaps.MapError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path)
