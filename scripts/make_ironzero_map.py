#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the fixed IronZero 25v25 tank arena map from the real Artillery terrain.

Stage-1 arena for the real-game PPO training path:

* terrain is the untouched ``maps/challenge/Artillery.tmx`` layout (Ground /
  Items / Triggers tileset references are preserved),
* the player army (team 0) and the AI army (team 1) are exactly 25 ``tank``
  units each (no buildings, no builders, no economy),
* both spawn formations are an exact 180-degree rotation about the map centre
  (``(x, y) -> (width_px - x, height_px - y)``), so neither side gets a
  positional advantage,
* every spawn tile and its 8 neighbours are passable ground for LAND movement,
  and the minimum spacing between two tanks is >= 24 px (actual 40 px),
* ``Triggers`` is rewritten to ``type=mission`` / ``winCondition=none`` (no
  auto victory) and ``credits=0`` on both team infos; team 1 keeps
  ``basicAI=true`` so the built-in AI is aggressive,
* a Python-side terrain grid (``blocked`` mask) is exported with the same
  schema the sandbox backend produces (``TerrainGrid.from_payload``).

The output is deterministic and idempotent: every byte is derived from the
source TMX (gzip mtime is pinned to 0), so re-running the script on the same
assets reproduces identical files (useful for git and for reproducibility).

Passability rule (reverse-engineered from the real 1.15 engine, see
``MapTile.createTile`` / ``MapCosts.rebuildTerrainCost`` in game-lib-src):

* Ground tile properties that make a tile unpassable for ``MovementType.LAND``:
  ``water``, ``cliff``, ``cliff-soft``, ``trees``, ``large-cliff``,
  ``large-rock``, ``block-land``, ``lava``, ``lava-cliff``.
* ``Shallow Water`` tiles carry the ``water`` property in 1.15, so they are
  **blocked** for tanks as well: tanks cannot wade shallow water in the real
  engine (the 5.6% shallow-water samples in old platform notes came from
  water/hover/air units, not tanks).
* ``small-rock`` sets ``blockLevel = 40``: passable, just expensive, so it is
  NOT blocked. ``large-rock`` sets ``blockLevel = -1`` and is blocked.
* The ``tree`` (singular) property is ignored by the 1.15 parser (only
  ``trees`` / ``large-cliff`` set the blocking flag).
* Item-layer (``Items``) tiles additionally block LAND movement when they carry
  ``res_pool`` (resource pools) or one of the tree/rock blocking properties, or
  a negative ``blockLevel``; those positions are folded into the same mask.

Usage::

    python scripts/make_ironzero_map.py
    python scripts/make_ironzero_map.py --runtime .runtime/rwpp-runtime-1.15
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import struct
import sys
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME = REPO_ROOT / ".runtime" / "rwpp-runtime-1.15"
SOURCE_MAP_REL = "maps/challenge/Artillery.tmx"
TARGET_MAP_REL = "maps/ironzero/IronZero25.tmx"

MAP_ID = "IronZero25"
MAP_NAME = "IronZero25"
TANK_UNIT_NAME = "tank"
UNITS_TILESET_NAME = "units"
TEAM_SIZES = {0: 25, 1: 25}

# Spawn formation: a 5x5 block with a 2-tile (40 px) step.  Every grid point is
# a tank; the AI/enemy block is the 180-degree rotation of the player block.
GRID = 5
GRID_STEP_TILES = 2
SPAWN_CLEARANCE_TILES = 1  # every 8-neighbour of a spawn tile must be passable
MIN_UNIT_SPACING_PX = 24.0
MIN_TEAM_SEPARATION_PX = 700.0
TARGET_CENTER_OFFSET_PX = 600.0
CENTER_OFFSET_RANGE_PX = (450.0, 750.0)

# Ground properties that block LAND movement in 1.15 (MapTile + MapCosts).
GROUND_BLOCKING_PROPERTIES = frozenset(
    {
        "water",
        "cliff",
        "cliff-soft",
        "trees",
        "large-cliff",
        "large-rock",
        "block-land",
        "lava",
        "lava-cliff",
    }
)
# Item-layer properties that block LAND movement (MapCosts.rebuildTerrainCost).
ITEM_BLOCKING_PROPERTIES = frozenset(
    {"res_pool", "trees", "large-cliff", "large-rock", "block-land"}
)
BLOCKED_RULE_TEXT = (
    "blocked=1 for LAND movement. Ground: water|cliff|cliff-soft|trees|large-cliff|"
    "large-rock|block-land|lava|lava-cliff OR blockLevel<0. Items: res_pool|trees|"
    "large-cliff|large-rock|block-land OR blockLevel<0. Small water is NOT special: "
    "Shallow Water tiles carry the 'water' property, so LAND units (tank) cannot "
    "enter them in real 1.15. small-rock (blockLevel=40) is passable but costly; "
    "the singular 'tree' property is ignored by 1.15."
)


# ----------------------------------------------------------------------
# TMX / tileset parsing
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class TileSetInfo:
    first_gid: int
    name: str
    local_properties: dict[int, dict[str, str]]

    def contains(self, gid: int, next_first_gid: int | None) -> bool:
        if gid < self.first_gid:
            return False
        return next_first_gid is None or gid < next_first_gid


def _tile_properties(element: ET.Element) -> dict[int, dict[str, str]]:
    props: dict[int, dict[str, str]] = {}
    for tile in element.findall("tile"):
        values = {
            prop.get("name", ""): prop.get("value", "")
            for prop in tile.findall("properties/property")
        }
        if values:
            props[int(tile.get("id", "0"))] = values
    return props


def load_tilesets(map_root: ET.Element, assets_root: Path) -> list[TileSetInfo]:
    """Resolve every tileset referenced by the map (embedded or external .tsx)."""

    tilesets: list[TileSetInfo] = []
    for element in map_root.findall("tileset"):
        first_gid = int(element.get("firstgid", "1"))
        source = element.get("source")
        if source:
            tsx_path = assets_root / "tilesets" / source
            if not tsx_path.is_file():
                raise SystemExit(f"tileset not found: {tsx_path}")
            tsx_root = ET.parse(tsx_path).getroot()
            name = tsx_root.get("name", Path(source).stem)
            props = _tile_properties(tsx_root)
        else:
            name = element.get("name", "")
            props = _tile_properties(element)
        tilesets.append(TileSetInfo(first_gid=first_gid, name=name, local_properties=props))
    tilesets.sort(key=lambda ts: ts.first_gid)
    return tilesets


def resolve_gid(
    gid: int, tilesets: list[TileSetInfo]
) -> tuple[TileSetInfo, int, dict[str, str]] | None:
    raw = gid & ~0xF0000000  # strip Tiled flip flags
    if raw <= 0:
        return None
    chosen: TileSetInfo | None = None
    for index, tileset in enumerate(tilesets):
        next_first = tilesets[index + 1].first_gid if index + 1 < len(tilesets) else None
        if tileset.contains(raw, next_first):
            chosen = tileset
            break
    if chosen is None:
        return None
    local_id = raw - chosen.first_gid
    return chosen, local_id, chosen.local_properties.get(local_id, {})


def decode_layer(layer: ET.Element, width: int, height: int) -> list[int]:
    data = layer.find("data")
    if data is None:
        return [0] * (width * height)
    encoding = (data.get("encoding") or "").lower()
    compression = (data.get("compression") or "").lower()
    text = (data.text or "").strip()
    if encoding != "base64":
        raise SystemExit(f"unsupported layer encoding: {encoding!r} (expected base64)")
    raw = base64.b64decode(text)
    if compression == "gzip":
        raw = gzip.decompress(raw)
    elif compression:
        raise SystemExit(f"unsupported layer compression: {compression!r}")
    count = len(raw) // 4
    values = list(struct.unpack(f"<{count}I", raw[: count * 4]))
    values = [value & ~0xF0000000 for value in values]
    if len(values) < width * height:
        values.extend([0] * (width * height - len(values)))
    return values[: width * height]


def encode_layer(values: Iterable[int]) -> str:
    buffer = list(values)
    payload = struct.pack(f"<{len(buffer)}I", *buffer)
    # mtime=0 keeps the gzip stream deterministic across runs.
    return base64.b64encode(gzip.compress(payload, compresslevel=9, mtime=0)).decode("ascii")


def find_layer(root: ET.Element, name: str) -> ET.Element | None:
    for layer in root.findall("layer"):
        if (layer.get("name") or "").lower() == name.lower():
            return layer
    return None


def find_objectgroup(root: ET.Element, name: str) -> ET.Element | None:
    for group in root.findall("objectgroup"):
        if (group.get("name") or "").lower() == name.lower():
            return group
    return None


def find_object(group: ET.Element, name: str) -> ET.Element | None:
    for obj in group.findall("object"):
        if (obj.get("name") or "").lower() == name.lower():
            return obj
    return None


def set_property(obj: ET.Element, name: str, value: str) -> None:
    props = obj.find("properties")
    if props is None:
        props = ET.SubElement(obj, "properties")
    for prop in props.findall("property"):
        if prop.get("name") == name:
            prop.set("value", value)
            return
    ET.SubElement(props, "property", {"name": name, "value": value})


# ----------------------------------------------------------------------
# Terrain / spawn computation
# ----------------------------------------------------------------------
def _has_negative_block_level(props: dict[str, str]) -> bool:
    try:
        return int(props.get("blockLevel", "")) < 0
    except (TypeError, ValueError):
        return False


def ground_tile_blocked(props: dict[str, str]) -> bool:
    if not props:
        return False
    if props.keys() & GROUND_BLOCKING_PROPERTIES:
        return True
    return _has_negative_block_level(props)


def item_tile_blocked(props: dict[str, str]) -> bool:
    if not props:
        return False
    # Unit tiles on the Items layer are spawn markers, not terrain: they are
    # stripped out separately (strip_unit_tiles_from_items) and never part of
    # the blocked mask of the produced arena.
    if "unit" in props or "customUnit" in props:
        return False
    if props.keys() & ITEM_BLOCKING_PROPERTIES:
        return True
    return _has_negative_block_level(props)


def build_blocked_mask(
    ground: list[int],
    items: list[int],
    tilesets: list[TileSetInfo],
    width: int,
    height: int,
) -> tuple[list[int], dict[str, int]]:
    stats = {
        "ground_water_or_cliff": 0,
        "items_resource_or_blocking": 0,
        "total": 0,
    }
    mask = [0] * (width * height)
    for index, gid in enumerate(ground):
        info = resolve_gid(gid, tilesets)
        if info is not None and ground_tile_blocked(info[2]):
            mask[index] = 1
            stats["ground_water_or_cliff"] += 1
    for index, gid in enumerate(items):
        info = resolve_gid(gid, tilesets)
        if info is not None and item_tile_blocked(info[2]) and not mask[index]:
            mask[index] = 1
            stats["items_resource_or_blocking"] += 1
    stats["total"] = sum(mask)
    return mask, stats


def _passable(land: list[list[bool]], x: int, y: int, width: int, height: int) -> bool:
    return 0 <= x < width and 0 <= y < height and land[y][x]


def _clear(land: list[list[bool]], x: int, y: int, width: int, height: int, margin: int) -> bool:
    for dx in range(-margin, margin + 1):
        for dy in range(-margin, margin + 1):
            if not _passable(land, x + dx, y + dy, width, height):
                return False
    return True


def choose_spawns(
    land: list[list[bool]],
    width: int,
    height: int,
    tile_width: int,
    tile_height: int,
    team0_units: int = 25,
    team1_units: int = 25,
) -> dict[str, Any]:
    """Deterministically pick the best rotation-symmetric spawn pair."""

    centre_x = width * tile_width / 2.0
    centre_y = height * tile_height / 2.0
    best: tuple[float, int, int] | None = None  # (score, y0, x0)

    for y0 in range(1, height - (GRID - 1) * GRID_STEP_TILES):
        for x0 in range(1, width - (GRID - 1) * GRID_STEP_TILES):
            tiles = [
                (x0 + GRID_STEP_TILES * i, y0 + GRID_STEP_TILES * j)
                for j in range(GRID)
                for i in range(GRID)
            ]
            if not all(
                _clear(land, x, y, width, height, SPAWN_CLEARANCE_TILES) for x, y in tiles
            ):
                continue
            mirrored = [(width - 1 - x, height - 1 - y) for x, y in tiles]
            if not all(
                _clear(land, x, y, width, height, SPAWN_CLEARANCE_TILES) for x, y in mirrored
            ):
                continue
            spawn_centre = (
                (x0 + GRID_STEP_TILES * (GRID - 1) / 2.0) * tile_width + tile_width / 2.0,
                (y0 + GRID_STEP_TILES * (GRID - 1) / 2.0) * tile_height + tile_height / 2.0,
            )
            offset = (
                (spawn_centre[0] - centre_x) ** 2 + (spawn_centre[1] - centre_y) ** 2
            ) ** 0.5
            if not CENTER_OFFSET_RANGE_PX[0] <= offset <= CENTER_OFFSET_RANGE_PX[1]:
                continue
            positions = [
                (x * tile_width + tile_width / 2.0, y * tile_height + tile_height / 2.0)
                for x, y in tiles
            ]
            mirrored_positions = [
                (width * tile_width - px, height * tile_height - py)
                for px, py in positions
            ]
            smallest = min(
                ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
                for ax, ay in positions
                for bx, by in mirrored_positions
            )
            if smallest < MIN_TEAM_SEPARATION_PX:
                continue
            score = abs(offset - TARGET_CENTER_OFFSET_PX)
            if best is None or (score, y0, x0) < best:
                best = (score, y0, x0)

    if best is None:
        raise SystemExit(
            "no rotation-symmetric spawn block found; relax MIN_TEAM_SEPARATION_PX/"
            "CENTER_OFFSET_RANGE_PX or spawn clearance"
        )

    _, y0, x0 = best
    full_team0_tiles = [
        (x0 + GRID_STEP_TILES * i, y0 + GRID_STEP_TILES * j)
        for j in range(GRID)
        for i in range(GRID)
    ]
    full_team1_tiles = [(width - 1 - x, height - 1 - y) for x, y in full_team0_tiles]
    full_team0_px = [
        (x * tile_width + tile_width / 2.0, y * tile_height + tile_height / 2.0)
        for x, y in full_team0_tiles
    ]
    full_team1_px = [(width * tile_width - x, height * tile_height - y) for x, y in full_team0_px]

    team0_units = max(1, min(len(full_team0_tiles), int(team0_units)))
    team1_units = max(1, min(len(full_team1_tiles), int(team1_units)))

    team0_tiles = full_team0_tiles[:team0_units]
    team1_tiles = full_team1_tiles[:team1_units]
    team0_px = full_team0_px[:team0_units]
    team1_px = full_team1_px[:team1_units]

    def _min_pairwise(points: list[tuple[float, float]]) -> float:
        if len(points) <= 1:
            return float("inf")
        return min(
            ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
            for i, (ax, ay) in enumerate(points)
            for bx, by in points[i + 1 :]
        )

    within = min(_min_pairwise(team0_px), _min_pairwise(team1_px))
    between = min(
        ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
        for ax, ay in team0_px
        for bx, by in team1_px
    )
    if within < MIN_UNIT_SPACING_PX and len(team0_px) > 1:
        raise SystemExit(f"unit spacing {within:.2f}px < {MIN_UNIT_SPACING_PX}px")

    def _summary(tiles: list[tuple[int, int]], points: list[tuple[float, float]]) -> dict[str, Any]:
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return {
            "center_px": [round((min(xs) + max(xs)) / 2.0, 1), round((min(ys) + max(ys)) / 2.0, 1)],
            "bbox_px": [min(xs), min(ys), max(xs), max(ys)],
            "bbox_tile": [
                min(tx for tx, _ in tiles),
                min(ty for _, ty in tiles),
                max(tx for tx, _ in tiles),
                max(ty for _, ty in tiles),
            ],
            "tiles": [[tx, ty] for tx, ty in tiles],
            "positions_px": [[round(px, 1), round(py, 1)] for px, py in points],
        }

    return {
        "grid": {"rows": GRID, "cols": GRID, "step_tiles": GRID_STEP_TILES,
                 "step_px": GRID_STEP_TILES * tile_width},
        "center_px": [round(centre_x, 1), round(centre_y, 1)],
        "center_offset_px": round(((full_team0_px[12][0] - centre_x) ** 2 + (full_team0_px[12][1] - centre_y) ** 2) ** 0.5, 1),
        "spacing": {
            "min_within_team_px": round(within if within != float("inf") else 0.0, 2),
            "min_between_teams_px": round(between, 2),
            "required_min_px": MIN_UNIT_SPACING_PX,
        },
        "team_0": _summary(team0_tiles, team0_px),
        "team_1": _summary(team1_tiles, team1_px),
    }


# ----------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------
def build_units_layer_data(
    unit_gids: dict[int, int],
    spawns: dict[str, Any],
    width: int,
    height: int,
) -> list[int]:
    data = [0] * (width * height)
    for team, key in ((0, "team_0"), (1, "team_1")):
        gid = unit_gids[team]
        for tx, ty in spawns[key]["tiles"]:
            index = ty * width + tx
            if data[index] != 0:
                raise SystemExit(f"duplicate unit spawn at tile ({tx},{ty})")
            data[index] = gid
    return data


def strip_unit_tiles_from_items(
    items: list[int], tilesets: list[TileSetInfo]
) -> tuple[list[int], int]:
    cleaned = list(items)
    removed = 0
    for index, gid in enumerate(cleaned):
        info = resolve_gid(gid, tilesets)
        if info is not None and ("unit" in info[2] or "customUnit" in info[2]):
            cleaned[index] = 0
            removed += 1
    return cleaned, removed


def _tank_gids(tilesets: list[TileSetInfo], width: int, height: int) -> dict[int, int]:
    gids: dict[int, int] = {}
    for tileset in tilesets:
        if tileset.name.lower() != UNITS_TILESET_NAME:
            continue
        for local_id, props in tileset.local_properties.items():
            if props.get("unit") != TANK_UNIT_NAME:
                continue
            team = props.get("team")
            if team is not None and int(team) in TEAM_SIZES:
                gids[int(team)] = tileset.first_gid + local_id
    missing = [team for team in TEAM_SIZES if team not in gids]
    if missing:
        raise SystemExit(f"tank unit tiles missing for team(s): {missing}")
    return gids


def rewrite_triggers(root: ET.Element) -> dict[str, Any]:
    group = find_objectgroup(root, "Triggers")
    if group is None:
        raise SystemExit("map has no Triggers objectgroup")
    map_info = find_object(group, "map_info")
    if map_info is None:
        raise SystemExit("Triggers has no map_info object")
    set_property(map_info, "fog", "map")
    set_property(map_info, "type", "mission")
    set_property(map_info, "winCondition", "none")
    set_property(
        map_info,
        "introText",
        "IronZero arena: 25 tanks per side, no economy. Eliminate the enemy tanks.",
    )

    for team, key in ((0, "team_0_info"), (1, "team_1_info")):
        obj = find_object(group, key)
        if obj is None:
            raise SystemExit(f"Triggers has no {key} object")
        if obj.get("type") != "team_info":
            obj.set("type", "team_info")
        set_property(obj, "team", str(team))
        set_property(obj, "credits", "0")
        set_property(obj, "allyGroup", str(team))
    team1 = find_object(group, "team_1_info")
    assert team1 is not None
    set_property(team1, "basicAI", "true")

    return {
        "map_info": {"type": "mission", "winCondition": "none", "fog": "map"},
        "team_0_info": {"team": 0, "credits": 0, "allyGroup": 0},
        "team_1_info": {"team": 1, "credits": 0, "allyGroup": 1, "basicAI": True},
    }


def _replace_layer_data(layer: ET.Element, values: list[int]) -> None:
    data = layer.find("data")
    if data is None:
        raise SystemExit(f"layer {layer.get('name')!r} has no <data>")
    data.set("encoding", "base64")
    data.set("compression", "gzip")
    data.text = "\n   " + encode_layer(values) + "\n  "


def generate(args: argparse.Namespace) -> int:
    runtime = Path(args.runtime).resolve()
    assets_root = runtime / "assets"
    source_path = (assets_root / SOURCE_MAP_REL) if not args.source_map else Path(args.source_map).resolve()
    if not source_path.is_file():
        raise SystemExit(f"source map not found: {source_path}")
    if not (assets_root / "tilesets").is_dir():
        raise SystemExit(f"tilesets not found under {assets_root / 'tilesets'}")

    out_tmx = Path(args.out_tmx).resolve()
    out_terrain = Path(args.out_terrain).resolve()
    out_meta = Path(args.out_meta).resolve()
    install_path = runtime / "assets" / TARGET_MAP_REL

    tree = ET.parse(source_path)
    root = tree.getroot()
    if root.tag != "map":
        raise SystemExit(f"not a TMX map: {source_path}")
    width = int(root.get("width", "0"))
    height = int(root.get("height", "0"))
    tile_width = int(root.get("tilewidth", "20"))
    tile_height = int(root.get("tileheight", "20"))
    if min(width, height, tile_width, tile_height) <= 0:
        raise SystemExit("map has invalid dimensions")

    tilesets = load_tilesets(root, assets_root)

    ground_layer = find_layer(root, "Ground")
    items_layer = find_layer(root, "Items")
    units_layer = find_layer(root, "Units")
    if ground_layer is None or items_layer is None:
        raise SystemExit("source map must contain Ground and Items layers")
    if units_layer is None:
        raise SystemExit("source map must contain a Units layer")

    ground = decode_layer(ground_layer, width, height)
    items = decode_layer(items_layer, width, height)

    blocked, blocked_stats = build_blocked_mask(ground, items, tilesets, width, height)
    land = [[not blocked[y * width + x] for x in range(width)] for y in range(height)]

    # Sanity: the whole walkable area must form a single connected component so
    # the two armies can actually reach each other (no island spawns).
    started = False
    seen = [[False] * width for _ in range(height)]
    largest = 0
    for y in range(height):
        for x in range(width):
            if not land[y][x] or seen[y][x]:
                continue
            queue = deque([(x, y)])
            seen[y][x] = True
            size = 0
            while queue:
                cx, cy = queue.popleft()
                size += 1
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < width and 0 <= ny < height and land[ny][nx] and not seen[ny][nx]:
                        seen[ny][nx] = True
                        queue.append((nx, ny))
            largest = max(largest, size)
    walkable = sum(1 for row in land for value in row if value)
    if largest != walkable:
        raise SystemExit(
            f"walkable ground is not one connected component (largest={largest}/{walkable})"
        )

    team0_units = int(getattr(args, "team0_units", 25))
    team1_units = int(getattr(args, "team1_units", team0_units))
    map_name_str = str(getattr(args, "map_name", "") or "")
    if not map_name_str:
        map_name_str = f"IronZero{team0_units}" if team0_units == team1_units else f"IronZero_{team0_units}v{team1_units}"
    map_id_str = map_name_str

    target_map_rel = f"maps/ironzero/{map_name_str}.tmx"
    install_path = runtime / "assets" / "maps" / "ironzero" / f"{map_name_str}.tmx"

    spawns = choose_spawns(
        land,
        width,
        height,
        tile_width,
        tile_height,
        team0_units=team0_units,
        team1_units=team1_units,
    )
    unit_gids = _tank_gids(tilesets, width, height)
    units_data = build_units_layer_data(unit_gids, spawns, width, height)
    cleaned_items, removed_item_units = strip_unit_tiles_from_items(items, tilesets)

    triggers = rewrite_triggers(root)
    # Ground (and the skipped "set" layer) keep their original tile data byte
    # for byte; only Items (unit tiles removed) and Units (new arena) change.
    _replace_layer_data(items_layer, cleaned_items)
    _replace_layer_data(units_layer, units_data)

    body = ET.tostring(root, encoding="unicode")
    tmx_text = '<?xml version="1.0" encoding="UTF-8"?>\n' + body + "\n"

    blocked_count = sum(blocked)
    total_tiles = width * height
    terrain_payload = {
        "id": map_id_str,
        "name": map_name_str,
        "width": width,
        "height": height,
        "tile_width": tile_width,
        "tile_height": tile_height,
        "width_px": width * tile_width,
        "height_px": height * tile_height,
        "blocked": blocked,
    }

    meta = {
        "map_id": map_id_str,
        "map_name": map_name_str,
        "source_map": SOURCE_MAP_REL,
        "target_map": target_map_rel,
        "runtime_assets": str(assets_root),
        "size": {
            "width": width,
            "height": height,
            "tile_width": tile_width,
            "tile_height": tile_height,
            "width_px": width * tile_width,
            "height_px": height * tile_height,
        },
        "unit_counts": {
            "team_0": team0_units,
            "team_1": team1_units,
            "total": team0_units + team1_units,
        },
        "unit_type": TANK_UNIT_NAME,
        "tank_tile": {
            "tileset": UNITS_TILESET_NAME,
            "tileset_firstgid": unit_gids[0] - _tank_local_id(tilesets, 0),
            "team_0_local_tile_id": _tank_local_id(tilesets, 0),
            "team_0_gid": unit_gids[0],
            "team_1_local_tile_id": _tank_local_id(tilesets, 1),
            "team_1_gid": unit_gids[1],
        },
        "spawn": spawns,
        "symmetry": {
            "kind": "180_degree_rotation_about_map_center",
            "rule": "(x, y) -> (width_px - x, height_px - y); team_1 = rotate(team_0)",
            "center_px": spawns["center_px"],
        },
        "terrain": {
            "blocked_tiles": blocked_count,
            "walkable_tiles": total_tiles - blocked_count,
            "total_tiles": total_tiles,
            "blocked_ratio": round(blocked_count / total_tiles, 5),
            "walkable_components": 1,
            "blocked_rule": BLOCKED_RULE_TEXT,
            "blocked_stats": blocked_stats,
        },
        "triggers": triggers,
        "artifacts": {
            "tmx": str(out_tmx),
            "installed_tmx": str(install_path),
            "terrain_json": str(out_terrain),
            "meta_json": str(out_meta),
        },
        "generator": "scripts/make_ironzero_map.py",
    }

    for path, text in ((out_tmx, tmx_text), (install_path, tmx_text)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    out_terrain.parent.mkdir(parents=True, exist_ok=True)
    out_terrain.write_text(
        json.dumps(terrain_payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"map            : {out_tmx}")
    print(f"installed      : {install_path}")
    print(f"terrain        : {out_terrain}")
    print(f"meta           : {out_meta}")
    print(f"size           : {width}x{height} tiles = {width * tile_width}x{height * tile_height} px")
    print(f"blocked        : {blocked_count}/{total_tiles} = {blocked_count / total_tiles:.4f}")
    print(f"tank gids      : team0={unit_gids[0]} team1={unit_gids[1]}")
    t0 = spawns["team_0"]["center_px"]
    t1 = spawns["team_1"]["center_px"]
    print(f"spawn centers  : team0={t0} team1={t1} (centre={spawns['center_px']})")
    print(
        "spacing        : within={:.1f}px between={:.1f}px (min required {:.0f}px)".format(
            spawns["spacing"]["min_within_team_px"],
            spawns["spacing"]["min_between_teams_px"],
            MIN_UNIT_SPACING_PX,
        )
    )
    print(f"items units    : removed {removed_item_units} unit tile(s) from Items layer")
    return 0


def _tank_local_id(tilesets: list[TileSetInfo], team: int) -> int:
    for tileset in tilesets:
        if tileset.name.lower() != UNITS_TILESET_NAME:
            continue
        for local_id, props in tileset.local_properties.items():
            if props.get("unit") == TANK_UNIT_NAME and str(props.get("team")) == str(team):
                return local_id
    raise SystemExit(f"tank tile for team {team} not found")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the IronZero tank arena map")
    parser.add_argument("--runtime", default=str(DEFAULT_RUNTIME), help="RWPP runtime directory")
    parser.add_argument("--source-map", default="", help="override source TMX path")
    parser.add_argument("--team0-units", type=int, default=25, help="player tank count (1..25)")
    parser.add_argument("--team1-units", type=int, default=25, help="enemy tank count (1..25)")
    parser.add_argument("--map-name", default="", help="map name override (e.g. IronZero10)")
    parser.add_argument(
        "--out-tmx", default=""
    )
    parser.add_argument(
        "--out-terrain", default=""
    )
    parser.add_argument(
        "--out-meta", default=""
    )
    return parser


def ensure_arena_map(
    team0_units: int = 25,
    team1_units: int | None = None,
    runtime: Path | str | None = None,
) -> tuple[Path, str, str]:
    """Ensure an IronZero arena map with the requested unit counts exists across all runtimes.

    Returns (tmx_path, map_relative_path, map_name).
    """

    t0 = max(1, min(25, int(team0_units)))
    t1 = max(1, min(25, int(team1_units if team1_units is not None else t0)))
    name = f"IronZero{t0}" if t0 == t1 else f"IronZero_{t0}v{t1}"
    rel = f"maps/ironzero/{name}.tmx"
    out_tmx = REPO_ROOT / "configs" / "maps" / f"{name}.tmx"
    out_terrain = REPO_ROOT / "configs" / "maps" / f"{name}.terrain.json"
    out_meta = REPO_ROOT / "configs" / "maps" / f"{name}.meta.json"

    rt = Path(runtime).resolve() if runtime else DEFAULT_RUNTIME
    install_tmx = rt / "assets" / "maps" / "ironzero" / f"{name}.tmx"

    if not out_tmx.is_file() or not install_tmx.is_file():
        args = argparse.Namespace(
            runtime=str(rt),
            source_map="",
            out_tmx=str(out_tmx),
            out_terrain=str(out_terrain),
            out_meta=str(out_meta),
            team0_units=t0,
            team1_units=t1,
            map_name=name,
        )
        generate(args)

    # Sync to other known runtime copies: C:\IronZero and .game_cache
    for extra_root in (
        Path(r"C:\IronZero\rwpp-runtime-1.15"),
        REPO_ROOT / ".game_cache",
        Path(r"E:\IronZeroBuild\rwpp-runtime-1.15"),
    ):
        if extra_root.is_dir():
            target = extra_root / "assets" / "maps" / "ironzero" / f"{name}.tmx" if (extra_root / "assets").is_dir() else extra_root / "maps" / "ironzero" / f"{name}.tmx"
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(out_tmx.read_bytes())
            except OSError:
                pass

    return out_tmx, rel, name


def _cli_entry() -> int:
    args = build_parser().parse_args()
    t0 = args.team0_units
    t1 = args.team1_units
    name = args.map_name or (f"IronZero{t0}" if t0 == t1 else f"IronZero_{t0}v{t1}")
    if not args.out_tmx:
        args.out_tmx = str(REPO_ROOT / "configs" / "maps" / f"{name}.tmx")
    if not args.out_terrain:
        args.out_terrain = str(REPO_ROOT / "configs" / "maps" / f"{name}.terrain.json")
    if not args.out_meta:
        args.out_meta = str(REPO_ROOT / "configs" / "maps" / f"{name}.meta.json")
    return generate(args)


if __name__ == "__main__":
    sys.exit(_cli_entry())
