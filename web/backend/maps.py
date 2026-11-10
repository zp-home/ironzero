# SPDX-License-Identifier: Apache-2.0
"""Read real Rusted Warfare maps and tilesets for the sandbox renderer.

The platform keeps a local read-only copy of the game's ``assets/tilesets``,
``assets/maps`` and tank sprites in a git-ignored ``.game_cache/`` directory
(see ``web/scripts/prepare_assets.ps1``). This module parses Tiled ``.tmx``
maps, resolves their tileset atlases and computes a good "battle window" so
the dashboard can render a faithful in-game view.

No game binary is redistributed: the cache is local-only and git-ignored.
"""

from __future__ import annotations

import base64
import gzip
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GID_FLAG_MASK = 0x0FFFFFFF
WATER_TILESET_HINTS = ("water",)
WATER_NAMES = {"water", "deep water", "shallow water"}
BATTLE_WINDOW_PX = 1200  # rendered crop size in map pixels (arena is 1000)


class MapError(ValueError):
    """Raised when a map cannot be parsed."""


def _assets_root() -> Path:
    import os

    root = os.environ.get("IRONZERO_GAME_ASSETS", "")
    if root:
        return Path(root).resolve()
    return (Path(__file__).resolve().parents[2] / ".game_cache").resolve()


ASSETS_ROOT = _assets_root()


def assets_available() -> bool:
    return (ASSETS_ROOT / "maps").is_dir() and (ASSETS_ROOT / "tilesets").is_dir()


# ----------------------------------------------------------------------
# Map discovery
# ----------------------------------------------------------------------
def map_id_for(relative: str) -> str:
    return base64.urlsafe_b64encode(relative.encode("utf-8")).decode("ascii").rstrip("=")


def relative_for(map_id: str) -> str:
    padding = "=" * (-len(map_id) % 4)
    try:
        return base64.urlsafe_b64decode(map_id + padding).decode("utf-8")
    except Exception as exc:  # noqa: BLE001
        raise MapError(f"invalid map id: {map_id}") from exc


_map_list_memory_cache: list[dict[str, Any]] | None = None


def list_maps(force: bool = False) -> list[dict[str, Any]]:
    """Enumerate all valid TMX maps in the asset cache, sorted by path."""
    global _map_list_memory_cache
    if _map_list_memory_cache is not None and not force:
        return _map_list_memory_cache

    maps_root = ASSETS_ROOT / "maps"
    if not maps_root.is_dir():
        return []
    items: list[dict[str, Any]] = []
    for path in sorted(maps_root.rglob("*.tmx")):
        relative = path.relative_to(ASSETS_ROOT).as_posix()
        info = _inspect_map_file(path)
        if info is None:
            continue
        name = path.stem
        players = 0
        if name.startswith("[p"):
            digits = ""
            for char in name[2:]:
                if char.isdigit():
                    digits += char
                else:
                    break
            players = int(digits) if digits else 0
        items.append(
            {
                "id": map_id_for(relative),
                "name": name,
                "relative": relative,
                "folder": path.relative_to(maps_root).parent.as_posix(),
                "width_tiles": info["width_tiles"],
                "height_tiles": info["height_tiles"],
                "tile_width": info["tile_width"],
                "tile_height": info["tile_height"],
                "width_px": info["width_tiles"] * info["tile_width"],
                "height_px": info["height_tiles"] * info["tile_height"],
                "players": players,
                "broken": info["broken"],
            }
        )
    _map_list_memory_cache = items
    return items


def _inspect_map_file(path: Path) -> dict[str, Any] | None:
    """Read map header and check for missing external tilesets in a single XML pass."""
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return None
    if root.tag != "map":
        return None
    broken = False
    for element in root.findall("tileset"):
        source = element.get("source")
        if source and not (ASSETS_ROOT / "tilesets" / source).is_file():
            broken = True
            break
    return {
        "width_tiles": int(root.get("width", "0")),
        "height_tiles": int(root.get("height", "0")),
        "tile_width": int(root.get("tilewidth", "20")),
        "tile_height": int(root.get("tileheight", "20")),
        "broken": broken,
    }


def _read_map_header(path: Path) -> dict[str, int]:
    info = _inspect_map_file(path)
    if info is None:
        raise MapError(f"not a map: {path}")
    return info


# ----------------------------------------------------------------------
# Map loading
# ----------------------------------------------------------------------
def load_map(map_id: str) -> dict[str, Any]:
    relative = relative_for(map_id)
    path = (ASSETS_ROOT / relative).resolve()
    if ASSETS_ROOT not in path.parents or not path.is_file():
        raise MapError(f"map not found: {relative}")
    root = ET.parse(path).getroot()
    tile_width = int(root.get("tilewidth", "20"))
    tile_height = int(root.get("tileheight", "20"))
    width = int(root.get("width", "0"))
    height = int(root.get("height", "0"))

    tilesets: list[dict[str, Any]] = []
    for element in root.findall("tileset"):
        tilesets.append(_parse_tileset(element, tile_width, tile_height))
    tilesets.sort(key=lambda item: item["firstgid"])

    layers: list[dict[str, Any]] = []
    ground: list[int] | None = None
    for element in root.findall("layer"):
        name = element.get("name", "")
        data = _decode_layer(element, width, height)
        layers.append({"name": name, "data": data})
        if name.lower() == "ground" and ground is None:
            ground = data

    water_mask = _water_mask(ground or [], tilesets, width, height)
    crop = _best_crop(water_mask, width, height, tile_width, tile_height, BATTLE_WINDOW_PX)
    land_ratio = 1.0 - (sum(water_mask) / max(1, len(water_mask)))

    return {
        "id": map_id,
        "name": path.stem,
        "relative": relative,
        "width": width,
        "height": height,
        "tile_width": tile_width,
        "tile_height": tile_height,
        "width_px": width * tile_width,
        "height_px": height * tile_height,
        "tilesets": tilesets,
        "layers": layers,
        "crop": crop,
        "land_ratio": round(land_ratio, 4),
    }


def _resolve_image(image_source: str, base_dir: Path | None = None) -> Path:
    """Resolve a tileset image path, tolerating Rusted Warfare's layout quirks."""

    if not image_source:
        raise MapError("tileset image has no source")
    source = image_source.replace("\\", "/")
    candidates: list[Path] = []
    if base_dir is not None:
        candidates.append((base_dir / source).resolve())
    candidates.append((ASSETS_ROOT / "tilesets" / source).resolve())
    candidates.append((ASSETS_ROOT / source).resolve())
    # Embedded tilesets sometimes carry a bogus leading folder ("terrain/bitmaps/x.png"
    # while the atlas actually lives in assets/tilesets/bitmaps/x.png).
    parts = source.split("/")
    if len(parts) > 1:
        candidates.append((ASSETS_ROOT / "tilesets" / "/".join(parts[1:])).resolve())
    candidates.append((ASSETS_ROOT / "tilesets" / parts[-1]).resolve())
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = sorted((ASSETS_ROOT / "tilesets").rglob(Path(source).name))
    if matches:
        return matches[0].resolve()
    raise MapError(f"tileset image not found: {image_source}")


def _parse_tileset(element: ET.Element, tile_width: int, tile_height: int) -> dict[str, Any]:
    firstgid = int(element.get("firstgid", "1"))
    source = element.get("source")
    if source:
        tsx_path = (ASSETS_ROOT / "tilesets" / source).resolve()
        if not tsx_path.is_file():
            raise MapError(f"tileset not found: {source}")
        tileset_root = ET.parse(tsx_path).getroot()
        image_root = tileset_root.find("image")
        image_source = image_root.get("source", "") if image_root is not None else ""
        name = tileset_root.get("name", Path(source).stem)
        columns = tileset_root.get("columns")
        tilecount = tileset_root.get("tilecount")
        image_path = _resolve_image(image_source, base_dir=tsx_path.parent)
    else:
        image_root = element.find("image")
        if image_root is None:
            raise MapError("embedded tileset without image")
        image_source = image_root.get("source", "")
        name = element.get("name", Path(image_source).stem)
        columns = element.get("columns")
        tilecount = element.get("tilecount")
        image_path = _resolve_image(image_source)
    if not image_path.is_file():
        raise MapError(f"tileset image not found: {image_path}")
    image_width, image_height = _png_size(image_path)
    cols = int(columns) if columns else max(1, image_width // tile_width)
    rows = max(1, image_height // tile_height)
    return {
        "firstgid": firstgid,
        "name": name,
        "image": image_path.relative_to(ASSETS_ROOT).as_posix(),
        "columns": cols,
        "rows": rows,
        "tile_width": tile_width,
        "tile_height": tile_height,
        "tile_count": int(tilecount) if tilecount else cols * rows,
        "image_width": image_width,
        "image_height": image_height,
        "water": name.lower() in WATER_NAMES,
    }


def _decode_layer(element: ET.Element, width: int, height: int) -> list[int]:
    data = element.find("data")
    if data is None:
        return [0] * (width * height)
    encoding = (data.get("encoding") or "").lower()
    compression = (data.get("compression") or "").lower()
    text = (data.text or "").strip()
    if encoding == "base64":
        raw = base64.b64decode(text)
        if compression == "gzip":
            raw = gzip.decompress(raw)
        count = len(raw) // 4
        values = list(struct.unpack(f"<{count}I", raw[: count * 4]))
    elif encoding == "csv":
        values = [int(item) for item in text.replace("\n", "").split(",") if item.strip()]
    else:
        raise MapError(f"unsupported layer encoding: {encoding!r}")
    values = [value & GID_FLAG_MASK for value in values]
    if len(values) < width * height:
        values.extend([0] * (width * height - len(values)))
    return values[: width * height]


def _water_mask(
    ground: list[int],
    tilesets: list[dict[str, Any]],
    width: int,
    height: int,
) -> list[bool]:
    by_firstgid = sorted(tilesets, key=lambda item: item["firstgid"])
    mask = [False] * (width * height)
    for index, gid in enumerate(ground):
        if gid <= 0:
            continue
        tileset = None
        for candidate in by_firstgid:
            if gid >= candidate["firstgid"]:
                tileset = candidate
            else:
                break
        if tileset is not None and tileset.get("water"):
            mask[index] = True
    return mask


def _best_crop(
    water_mask: list[bool],
    width: int,
    height: int,
    tile_width: int,
    tile_height: int,
    window_px: int,
) -> dict[str, int]:
    window_w = min(width, max(1, window_px // tile_width))
    window_h = min(height, max(1, window_px // tile_height))
    best = (0.0, 0, 0)
    step = max(1, min(window_w, window_h) // 6)
    for y in range(0, height - window_h + 1, step):
        for x in range(0, width - window_w + 1, step):
            land = 0
            for row in range(y, y + window_h):
                base = row * width
                land += sum(1 for column in range(x, x + window_w) if not water_mask[base + column])
            ratio = land / (window_w * window_h)
            if ratio > best[0]:
                best = (ratio, x, y)
    ratio, x, y = best
    return {
        "x": x * tile_width,
        "y": y * tile_height,
        "width": window_w * tile_width,
        "height": window_h * tile_height,
        "tile_x": x,
        "tile_y": y,
        "tile_w": window_w,
        "tile_h": window_h,
        "land_ratio": round(ratio, 4),
    }


# ----------------------------------------------------------------------
# Asset files
# ----------------------------------------------------------------------
def resolve_asset(relative: str) -> Path:
    """Resolve a path relative to the game asset cache root (validated)."""

    relative = relative.replace("\\", "/").lstrip("/")
    path = (ASSETS_ROOT / relative).resolve()
    if ASSETS_ROOT not in path.parents or not path.is_file():
        raise MapError(f"asset not found: {relative}")
    return path


def blocked_mask(payload: dict[str, Any]) -> list[int]:
    """1 = impassable tile for the terrain simulation, row-major.

    Rusted Warfare tanks can wade through *shallow* water, so only ``water`` and
    ``deep water`` tiles are treated as blocked; this matches what the real
    engine does (observed on the Artillery challenge map).
    """

    width = int(payload.get("width", 0))
    height = int(payload.get("height", 0))
    blocked = [0] * (width * height)
    ground = None
    for layer in payload.get("layers", []):
        if str(layer.get("name", "")).lower() == "ground":
            ground = layer.get("data")
            break
    if ground is None:
        return blocked
    tilesets = sorted(payload.get("tilesets", []), key=lambda item: item["firstgid"])
    for index, gid in enumerate(ground[: width * height]):
        if gid <= 0:
            continue
        tileset = None
        for candidate in tilesets:
            if gid >= candidate["firstgid"]:
                tileset = candidate
            else:
                break
        if tileset is None or not tileset.get("water"):
            continue
        name = str(tileset.get("name", "")).lower()
        if "shallow" in name:
            continue  # passable in the real game
        blocked[index] = 1
    return blocked


def unit_sprites() -> dict[str, str]:
    units_dir = ASSETS_ROOT / "units"
    if not units_dir.is_dir():
        return {}
    sprites: dict[str, str] = {}
    
    # 优先使用游戏原版实体贴图
    mapping = {
        "tank": "units/tanks/tank.png",
        "tank_turret": "units/tanks/tank_turret.png",
        "command_center": "units/original/base.png" if (ASSETS_ROOT / "units/original/base.png").is_file() else "units/outpost/outpost.png",
        "builder": "units/original/builder.png" if (ASSETS_ROOT / "units/original/builder.png").is_file() else "units/combat_engineer/base.png",
        "factory": "units/original/land_factory.png" if (ASSETS_ROOT / "units/original/land_factory.png").is_file() else "units/mech_factory/mechFactoryTop.png",
        "extractor": "units/original/extractor.png" if (ASSETS_ROOT / "units/original/extractor.png").is_file() else "units/extractor/extractor.png",
        "turret_base": "units/original/turret_base.png" if (ASSETS_ROOT / "units/original/turret_base.png").is_file() else "units/turrets/turret_base.png",
        "turret_top": "units/original/turret_top.png" if (ASSETS_ROOT / "units/original/turret_top.png").is_file() else "units/turrets/turret_top.png",
        "combat_engineer": "units/combat_engineer/base.png",
        "mech_factory": "units/mech_factory/mechFactoryTop.png",
        "outpost": "units/outpost/outpost.png",
    }
    for key, rel_path in mapping.items():
        if (ASSETS_ROOT / rel_path).is_file():
            sprites[key] = rel_path

    # 同时保留原有的 tanks/*.png 映射
    tank_folder = units_dir / "tanks"
    if tank_folder.is_dir():
        for path in sorted(tank_folder.glob("*.png")):
            sprites.setdefault(path.stem, f"units/tanks/{path.name}")

    return sprites


def _png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise MapError(f"not a PNG: {path}")
    width, height = struct.unpack(">II", header[16:24])
    return width, height


__all__ = [
    "ASSETS_ROOT",
    "MapError",
    "assets_available",
    "blocked_mask",
    "list_maps",
    "load_map",
    "map_id_for",
    "relative_for",
    "resolve_asset",
    "unit_sprites",
]
