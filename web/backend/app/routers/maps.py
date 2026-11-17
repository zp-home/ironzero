# SPDX-License-Identifier: Apache-2.0
"""Real-game assets: map list, one parsed map, tileset/unit sprite files.

Reads only the local ``.game_cache`` copy through ``web/backend/maps.py``.

``/api/maps/catalog`` is a different question from ``/api/maps``: the latter is
"what can the sandbox render" (parsed from the asset cache, every ``.tmx`` on
disk), the former is "which maps may a training run be pointed at, and how big is
each one in pixels" — the host service's own registry, forwarded verbatim. The
form needs the second one: the observation normalises every position by the map's
pixel size, so a wrong number there silently skews the whole episode instead of
failing.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from ... import maps as gamemaps
from ... import unit_sprites as gamesprites
from .rwpp import _rwpp_call

router = APIRouter()

_map_list_cache: dict[str, Any] = {"at": 0.0, "payload": None}


@router.get("/api/maps/catalog")
def api_map_catalog() -> dict[str, Any]:
    """The host service's map registry: nine stock 2p maps with real pixel sizes.

    Declared *before* ``/api/maps/{map_id}`` on purpose — FastAPI matches in
    declaration order, so this static path has to win over the parameterised one
    (otherwise ``catalog`` is parsed as a base64 map id and 404s).

    No caching: the host registry is a handful of constants, and a cached copy
    would keep serving stale sizes after the maps are re-measured.
    """

    return _rwpp_call("/maps", timeout=10.0)


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
