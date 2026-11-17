# SPDX-License-Identifier: Apache-2.0
"""Registry of the stock two-player skirmish maps (``[p2]*.tmx``).

Multi-map training needs one place that knows *which* maps exist and *how big*
they are, because the observation normalises every position by the map's pixel
size. Hard-coding ``2600x2500`` (the old default) on a 4000x3200 map silently
squashes every normalised coordinate into the left two-thirds of the space, and
nothing raises. This module is that single source of truth:

* :data:`TWO_PLAYER_MAPS` -- the nine stock ``[p2]`` maps as :class:`MapSpec`
  values with their pixel sizes written as constants, so nothing has to parse a
  ``.tmx`` just to know how wide the world is;
* :func:`map_spec` / :func:`map_specs` / :func:`resolve_map_size` -- lookup by
  short name or by map-relative path;
* :func:`build_terrain_payload` / :func:`ensure_terrain_json` -- the
  :class:`~ironzero.terrain.grid.TerrainGrid` blocked mask for a stock map,
  cached under ``configs/maps/<short name>.terrain.json``;
* :func:`start_points` -- both bases, read from the map's unit markers.

Water detection on stock maps
-----------------------------
This is the part that looks wrong until you know the format. Rusted Warfare
1.15's stock maps reference **external** tilesets (``source="terrain/Dirt.tsx"``)
and those ``.tsx`` files declare only an image -- they carry no per-tile
``<properties>`` at all. So the ``water``/``cliff`` properties that
``scripts/make_ironzero_map.py`` relies on simply do not exist here, and a
property-only rule reports 0% water on every stock map.

The signal that *is* present is the tileset **name** (or its source path):
``Shallow Water``, ``Deep Water``, ``ridges/Water DeepWater - Flat.tsx``,
``Sand ShallowWater - Ridge``, ... A tile is water when the tileset it was
placed from carries one of :data:`WATER_NAME_MARKERS`. That rule is not a
heuristic guess: it reproduces all nine published stock ratios exactly, e.g.
Lake ``25.02%`` (4229/16900), Beach landing ``60.8%``, Hills ``0.0%``.
:func:`measure_water_ratio` re-runs it so a test can catch a regression.

Blocked mask
------------
``blocked`` follows the same *rule* as :func:`scripts.make_ironzero_map.build_blocked_mask`
(Ground water/cliff + Items resource/rock blockers, ``unit``/``customUnit``
markers are spawn points rather than terrain), with the water signal widened to
the tileset-name rule described above. On top of that an empty Ground cell
counts as blocked, because a cell with no terrain is not walkable.
"""

from __future__ import annotations

import base64
import gzip
import json
import os
import struct
import sys
import xml.etree.ElementTree as ET
import zlib
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "BLOCKED_RULE_TEXT",
    "GROUND_BLOCKING_PROPERTIES",
    "ITEM_BLOCKING_PROPERTIES",
    "MapSpec",
    "TWO_PLAYER_MAPS",
    "WATER_NAME_MARKERS",
    "build_terrain_payload",
    "default_assets_root",
    "ensure_terrain_json",
    "map_spec",
    "map_specs",
    "measure_water_ratio",
    "resolve_map_size",
    "start_points",
]

# ``src/ironzero/rwpp/maps.py`` -> parents[0]=rwpp, [1]=ironzero, [2]=src,
# [3]=repository root. Kept as a local constant (not imported from
# ``.config``) so this registry stays stdlib-only and cannot be broken by an
# edit elsewhere in the package.
REPO_ROOT = Path(__file__).resolve().parents[3]
#: Where the generated per-map blocked masks are cached.
TERRAIN_DIR = REPO_ROOT / "configs" / "maps"

#: Tileset-label substrings that mark a tile as water / cliff. Rusted Warfare's
#: stock maps name their tilesets after the terrain they draw (``Shallow
#: Water``, ``ridges/Sand ShallowWater - Ridge.tsx``), which is the only signal
#: available: the referenced ``.tsx`` files declare images and no properties.
WATER_NAME_MARKERS = ("water", "shallow", "deep", "cliff")

#: Ground properties that block LAND movement in 1.15 (``MapTile``/``MapCosts``).
GROUND_BLOCKING_PROPERTIES = frozenset(
    {
        "water",
        "deepWater",
        "cliff",
        "cliff-soft",
        "trees",
        "large-cliff",
        "large-rock",
        "block-land",
        "blockLand",
        "blockSea",
        "blockHover",
        "blockBuilding",
        "blockUnits",
        "lava",
        "lava-cliff",
    }
)
#: Item-layer properties that block LAND movement (``MapCosts.rebuildTerrainCost``).
ITEM_BLOCKING_PROPERTIES = frozenset(
    {
        "res_pool",
        "trees",
        "large-cliff",
        "large-rock",
        "block-land",
        "blockLand",
        "blockSea",
        "blockHover",
        "blockBuilding",
        "water",
        "deepWater",
        "cliff",
        "lava",
        "lava-cliff",
    }
)

BLOCKED_RULE_TEXT = (
    "blocked=1 for LAND movement. Ground: tileset label contains one of "
    "'water|shallow|deep|cliff', or the tile carries "
    "water|deepWater|cliff|cliff-soft|trees|large-cliff|large-rock|block-land|lava|"
    "lava-cliff, or blockLevel<0; an empty Ground cell is blocked too. Items: "
    "res_pool|trees|large-cliff|large-rock|block-land|... or blockLevel<0, but a "
    "tile carrying unit/customUnit is a spawn marker, not terrain. The tileset-name "
    "clause exists because stock 1.15 maps use external .tsx files that declare no "
    "tile properties."
)

#: ``Ground`` layer name in a stock map.
GROUND_LAYER = "Ground"
#: ``Items`` layer name (resource pools, trees, rocks).
ITEMS_LAYER = "Items"
#: ``Units`` layer name (pre-placed units).
UNITS_LAYER = "Units"
#: Tiled's flip/rotate flags live in the high nibble of a gid and must be masked off.
_GID_FLAGS = 0xF0000000


# ----------------------------------------------------------------------
# Registry value type
# ----------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class MapSpec:
    """Everything the rest of the system needs to know about one stock map.

    ``width_px`` / ``height_px`` are the numbers the observation normalises
    positions by, so they are stored as constants rather than re-derived on
    every reset. ``tile_px`` is square: every stock ``[p2]`` map uses 20x20.
    """

    #: Short, human name used in configs and the dashboard, e.g. ``"Lake"``.
    name: str
    #: Path relative to the runtime's ``assets/`` directory.
    relative: str
    width_px: int
    height_px: int
    tile_px: int
    #: Fraction of the Ground layer that is water/cliff (0.0 .. 1.0).
    water_ratio: float
    #: True when the map has any water at all; sea factories are pointless
    #: otherwise, so callers branch on this instead of re-deriving it.
    has_water: bool
    #: Cached :class:`~ironzero.terrain.grid.TerrainGrid` payload for this map.
    terrain_path: Path
    #: Both bases as ``(team0, team1)`` pixel centres, empty when unknown.
    start_points: tuple[tuple[float, float], ...] = ()

    @property
    def width_tiles(self) -> int:
        return self.width_px // self.tile_px

    @property
    def height_tiles(self) -> int:
        return self.height_px // self.tile_px

    @property
    def tile_count(self) -> int:
        return self.width_tiles * self.height_tiles

    @property
    def size_px(self) -> tuple[float, float]:
        return (float(self.width_px), float(self.height_px))

    @property
    def file_name(self) -> str:
        """Last path component, e.g. ``[p2]Lake (2p).tmx``."""

        return self.relative.rsplit("/", 1)[-1]

    def base_of(self, team: int) -> tuple[float, float] | None:
        """The base of ``team`` (0 = us, 1 = enemy), or ``None`` if unknown."""

        if 0 <= team < len(self.start_points):
            return self.start_points[team]
        return None


def _spec(
    name: str,
    file_name: str,
    width_px: int,
    height_px: int,
    water_ratio: float,
    own: tuple[float, float],
    enemy: tuple[float, float],
    tile_px: int = 20,
    subdirectory: str = "skirmish",
) -> MapSpec:
    return MapSpec(
        name=name,
        relative=f"maps/{subdirectory}/{file_name}",
        width_px=width_px,
        height_px=height_px,
        tile_px=tile_px,
        water_ratio=water_ratio,
        has_water=water_ratio > 0.0,
        terrain_path=TERRAIN_DIR / f"{name}.terrain.json",
        start_points=(own, enemy),
    )


#: The nine stock ``[p2]`` skirmish maps, in the engine's own listing order.
#:
#: Sizes are the ``.tmx`` header values (``width``/``height`` tiles x 20 px);
#: ``water_ratio`` is the measured water share (see
#: :func:`measure_water_ratio`); ``start_points`` are the ``commandCenter``
#: markers from each map's ``Units`` layer, team 0 first. Every value here was
#: read out of the shipped 1.15 assets and cross-checked against the bridge's
#: first observation, which reports the same base coordinates.
TWO_PLAYER_MAPS: tuple[MapSpec, ...] = (
    _spec(
        "Beach landing",
        "[p2]Beach landing (2p) [by hxyy].tmx",
        4000,
        3200,
        0.6082,
        (3430.0, 1570.0),
        (570.0, 1390.0),
    ),
    _spec(
        "Big Island",
        "[p2]Big Island (2p).tmx",
        3600,
        3600,
        0.5067,
        (2990.0, 3070.0),
        (550.0, 490.0),
    ),
    _spec(
        "Dire_Straight",
        "[p2]Dire_Straight (2p) [by uber].tmx",
        2200,
        2200,
        0.4644,
        (1790.0, 1710.0),
        (470.0, 370.0),
    ),
    _spec(
        "Fire Bridge",
        "[p2]Fire Bridge (2p) [by uber].tmx",
        2400,
        2400,
        0.2233,
        (550.0, 2110.0),
        (1850.0, 310.0),
    ),
    _spec(
        "Hills",
        "[p2]Hills_(2p)_[By Tstis & KPSS].tmx",
        2600,
        2600,
        0.0,
        (270.0, 1290.0),
        (2330.0, 1290.0),
    ),
    _spec(
        "Ice Island",
        "[p2]Ice Island (2p).tmx",
        2900,
        2900,
        0.5156,
        (1270.0, 2390.0),
        (1390.0, 470.0),
    ),
    _spec(
        "Lake",
        "[p2]Lake (2p).tmx",
        2600,
        2600,
        0.2502,
        (990.0, 2010.0),
        (1230.0, 210.0),
    ),
    _spec(
        "Small_Island",
        "[p2]Small_Island (2p).tmx",
        2200,
        2200,
        0.6379,
        (990.0, 1670.0),
        (930.0, 370.0),
    ),
    _spec(
        "Two_cold_sides",
        "[p2]Two_cold_sides (2p).tmx",
        2700,
        2500,
        0.6159,
        (2430.0, 1470.0),
        (370.0, 1110.0),
    ),
)


# ----------------------------------------------------------------------
# Lookup
# ----------------------------------------------------------------------
def map_specs() -> tuple[MapSpec, ...]:
    """Every registered two-player map."""

    return TWO_PLAYER_MAPS


def _normalise_key(name_or_relative: str) -> str:
    key = str(name_or_relative).strip().replace("\\", "/")
    while key.startswith("./"):
        key = key[2:]
    return key


def _lookup_keys(spec: MapSpec) -> tuple[str, ...]:
    """Every string that should resolve to ``spec``, lower-cased."""

    file_name = spec.file_name
    stem = file_name[:-4] if file_name.lower().endswith(".tmx") else file_name
    return (
        spec.name.lower(),
        spec.relative.lower(),
        file_name.lower(),
        stem.lower(),
    )


def map_spec(name_or_relative: str) -> MapSpec:
    """Resolve a map by short name, file name or map-relative path.

    Accepts ``"Lake"``, ``"[p2]Lake (2p).tmx"`` and
    ``"maps/skirmish/[p2]Lake (2p).tmx"`` interchangeably, case-insensitively
    and with either slash direction, so callers can pass whatever they already
    have (a Web form value, a CLI arg, an asset manifest entry) without
    translating.

    Raises ``KeyError`` whose message lists the candidates: a typo in a config
    file should say what the legal values are, not just "not found".
    """

    key = _normalise_key(name_or_relative).lower()
    for spec in TWO_PLAYER_MAPS:
        if key in _lookup_keys(spec):
            return spec
    candidates = ", ".join(repr(spec.name) for spec in TWO_PLAYER_MAPS)
    raise KeyError(
        f"unknown map {name_or_relative!r}; known maps: {candidates} "
        f"(or a path ending in 'maps/skirmish/<file>.tmx')"
    )


def resolve_map_size(name_or_relative: str) -> tuple[float, float]:
    """``(width_px, height_px)`` for a map, as floats.

    The observation divides positions by the map size, so this is the function
    that keeps ``2600x2500`` from leaking into a 4000x3200 map.
    """

    return map_spec(name_or_relative).size_px


def default_assets_root() -> Path:
    """The RWPP runtime ``assets`` directory, resolved like the rest of the repo.

    ``IRONZERO_RWPP_RUNTIME`` wins, then the historical ``E:\\IronZeroBuild``
    build, then a repository-local runtime copy. Same precedence as
    ``scripts/make_ironzero_map.py`` so generated files and the trainer agree.
    """

    configured = os.environ.get("IRONZERO_RWPP_RUNTIME")
    if configured:
        return Path(configured) / "assets"
    build = Path(r"E:\IronZeroBuild\rwpp-runtime-1.15")
    if build.is_dir():
        return build / "assets"
    return REPO_ROOT / ".runtime" / "rwpp-runtime-1.15" / "assets"


# ----------------------------------------------------------------------
# .tmx parsing
# ----------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class _TileSet:
    """One ``<tileset>`` reference plus its resolved tile properties.

    ``properties`` is empty for the stock external ``.tsx`` tilesets (they
    declare images only); ``water_label`` is then the only thing that
    classifies their tiles.
    """

    first_gid: int
    label: str
    properties: dict[int, dict[str, str]]
    water_label: bool


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


def _is_water_label(*labels: str) -> bool:
    return any(
        marker in label.lower() for label in labels if label for marker in WATER_NAME_MARKERS
    )


def _tilesets(root: ET.Element, assets_root: Path) -> tuple[_TileSet, ...]:
    """Resolve every tileset the map references (embedded or external ``.tsx``).

    An external reference that cannot be found is not fatal -- ``.runtime``
    asset trees are copied around and some stock maps reference paths from the
    map author's machine (``../rustedWarfareMaps-testing/misc.tsx``). The label
    falls back to the ``.tsx`` stem, which still carries the name we key on.
    """

    tilesets: list[_TileSet] = []
    for element in root.findall("tileset"):
        first_gid = int(element.get("firstgid", "1"))
        source = element.get("source") or ""
        if source:
            tsx_path = assets_root / "tilesets" / source
            if tsx_path.is_file():
                tsx_root = ET.parse(tsx_path).getroot()
                name = tsx_root.get("name", Path(source).stem)
                properties = _tile_properties(tsx_root)
            else:
                name = Path(source).stem
                properties = {}
        else:
            name = element.get("name", "")
            properties = _tile_properties(element)
        tilesets.append(
            _TileSet(
                first_gid=first_gid,
                label=name,
                properties=properties,
                water_label=_is_water_label(name, source),
            )
        )
    tilesets.sort(key=lambda tileset: tileset.first_gid)
    return tuple(tilesets)


def _decode_layer(
    root: ET.Element, layer_name: str, width: int, height: int
) -> list[int]:
    """Decode a base64 layer (``gzip``/``zlib``/uncompressed/csv).

    Stock 1.15 maps use ``base64`` + ``gzip``; the fallbacks exist so a
    hand-edited or re-exported map does not turn into an exception deep inside
    the training loop.
    """

    for layer in root.findall("layer"):
        if (layer.get("name") or "").lower() != layer_name.lower():
            continue
        data = layer.find("data")
        if data is None:
            return [0] * (width * height)
        text = "".join((data.text or "").split())
        encoding = (data.get("encoding") or "").lower()
        if not text:
            return [0] * (width * height)
        if encoding != "base64":
            tokens = text.replace("\n", ",").split(",")
            numbers = [int(token) for token in tokens if token.strip().isdigit()]
            numbers = [value & ~_GID_FLAGS for value in numbers]
            return (numbers + [0] * (width * height))[: width * height]
        blob = base64.b64decode(text)
        compression = (data.get("compression") or "").lower()
        if compression == "gzip":
            blob = gzip.decompress(blob)
        elif compression == "zlib":
            blob = zlib.decompress(blob)
        elif compression:
            raise ValueError(
                f"unsupported layer compression {compression!r} in layer {layer_name!r}"
            )
        count = len(blob) // 4
        values = [
            value & ~_GID_FLAGS for value in struct.unpack(f"<{count}I", blob[: count * 4])
        ]
        if len(values) < width * height:
            values.extend([0] * (width * height - len(values)))
        return values[: width * height]
    return [0] * (width * height)


class _MapSource:
    """A parsed stock map: header geometry, tilesets and the three layers."""

    __slots__ = ("path", "width", "height", "tile_width", "tile_height", "_tilesets", "_root")

    def __init__(self, path: Path, assets_root: Path) -> None:
        self.path = path
        self._root = ET.parse(path).getroot()
        root = self._root
        if root.tag != "map":
            raise ValueError(f"not a TMX map: {path}")
        self.width = int(root.get("width", "0"))
        self.height = int(root.get("height", "0"))
        self.tile_width = int(root.get("tilewidth", "20"))
        self.tile_height = int(root.get("tileheight", "20"))
        if min(self.width, self.height, self.tile_width, self.tile_height) <= 0:
            raise ValueError(f"map {path} has invalid dimensions")
        self._tilesets = _tilesets(root, assets_root)

    @property
    def width_px(self) -> int:
        return self.width * self.tile_width

    @property
    def height_px(self) -> int:
        return self.height * self.tile_height

    def layer(self, name: str) -> list[int]:
        return _decode_layer(self._root, name, self.width, self.height)

    def resolve(self, gid: int) -> tuple[_TileSet, dict[str, str]] | None:
        """``(tileset, tile properties)`` for a gid, or ``None`` for gid 0."""

        raw = gid & ~_GID_FLAGS
        if raw <= 0:
            return None
        first_gids = [tileset.first_gid for tileset in self._tilesets]
        index = bisect_right(first_gids, raw) - 1
        if index < 0:
            return None
        tileset = self._tilesets[index]
        return tileset, tileset.properties.get(raw - tileset.first_gid, {})

    def is_water(self, gid: int) -> bool:
        resolved = self.resolve(gid)
        if resolved is None:
            return False
        tileset, props = resolved
        return tileset.water_label or bool(props.keys() & {"water", "deepWater", "cliff"})


def _open_map(spec: MapSpec | str, assets_root: Path | str | None) -> tuple[MapSpec, _MapSource]:
    resolved_spec = spec if isinstance(spec, MapSpec) else map_spec(spec)
    root = Path(assets_root) if assets_root is not None else default_assets_root()
    path = root / resolved_spec.relative
    if not path.is_file():
        raise FileNotFoundError(f"map file not found: {path}")
    return resolved_spec, _MapSource(path, root)


# ----------------------------------------------------------------------
# Terrain payload
# ----------------------------------------------------------------------
def _has_negative_block_level(props: dict[str, str]) -> bool:
    try:
        return int(props.get("blockLevel", "")) < 0
    except (TypeError, ValueError):
        return False


def _ground_blocked(tileset: _TileSet, props: dict[str, str]) -> bool:
    if tileset.water_label:
        return True
    if props.keys() & GROUND_BLOCKING_PROPERTIES:
        return True
    return _has_negative_block_level(props)


def _item_blocked(props: dict[str, str]) -> bool:
    if not props:
        return False
    # A tile carrying ``unit``/``customUnit`` is a pre-placed unit or a tree
    # sprite, i.e. a spawn marker rather than terrain: folding it into the mask
    # would make the two bases look impassable.
    if "unit" in props or "customUnit" in props:
        return False
    if props.keys() & ITEM_BLOCKING_PROPERTIES:
        return True
    return _has_negative_block_level(props)


def _blocked_mask(source: _MapSource, ground: list[int], items: list[int]) -> list[int]:
    width = source.width
    mask = [0] * (width * source.height)
    for index, gid in enumerate(ground):
        resolved = source.resolve(gid)
        if resolved is None:
            # No terrain at all: nothing can stand there.
            mask[index] = 1
            continue
        tileset, props = resolved
        if _ground_blocked(tileset, props):
            mask[index] = 1
    for index, gid in enumerate(items):
        if mask[index]:
            continue
        resolved = source.resolve(gid)
        if resolved is None:
            continue
        if _item_blocked(resolved[1]):
            mask[index] = 1
    return mask


def build_terrain_payload(
    spec: MapSpec | str, assets_root: Path | str | None = None
) -> dict[str, Any]:
    """The :class:`~ironzero.terrain.grid.TerrainGrid` payload for a stock map.

    The payload has exactly the keys ``TerrainGrid.from_payload`` reads, so it
    can be dumped to JSON and fed straight to ``TerrainGrid.from_file``.
    Geometry comes from the ``.tmx`` header, never from ``MapSpec``, so a
    mis-typed registry constant shows up as a mismatch instead of silently
    skewing every normalised coordinate.
    """

    resolved_spec, source = _open_map(spec, assets_root)
    ground = source.layer(GROUND_LAYER)
    items = source.layer(ITEMS_LAYER)
    blocked = _blocked_mask(source, ground, items)
    return {
        "id": resolved_spec.name,
        "name": resolved_spec.name,
        "width": source.width,
        "height": source.height,
        "tile_width": source.tile_width,
        "tile_height": source.tile_height,
        "width_px": source.width_px,
        "height_px": source.height_px,
        "blocked": blocked,
    }


def measure_water_ratio(spec: MapSpec | str, assets_root: Path | str | None = None) -> float:
    """Re-measure the Ground-layer water share of a stock map.

    Exposed so the water rule can be asserted against the published ratios: if
    someone "improves" :func:`_is_water_label`, ``Lake`` stops being 25.02% and
    a test fails instead of the training reward quietly changing meaning.
    """

    _resolved, source = _open_map(spec, assets_root)
    ground = source.layer(GROUND_LAYER)
    total = 0
    water = 0
    for gid in ground:
        if not gid:
            continue
        total += 1
        if source.is_water(gid):
            water += 1
    if total <= 0:
        return 0.0
    return water / total


def ensure_terrain_json(
    spec: MapSpec | str,
    assets_root: Path | str | None = None,
    out_dir: Path | str | None = None,
) -> Path:
    """Write (or reuse) ``configs/maps/<short name>.terrain.json`` and return it.

    The cache is keyed on modification time: as long as the ``.tmx`` is not
    newer than the JSON, the existing file is returned untouched. That keeps a
    training run from paying a ~170k-tile decode per map on every start while
    still refreshing after the assets are updated.
    """

    resolved_spec = spec if isinstance(spec, MapSpec) else map_spec(spec)
    root = Path(assets_root) if assets_root is not None else default_assets_root()
    map_path = root / resolved_spec.relative
    directory = Path(out_dir) if out_dir is not None else TERRAIN_DIR
    out_path = directory / f"{resolved_spec.name}.terrain.json"

    if out_path.is_file() and map_path.is_file():
        if out_path.stat().st_mtime >= map_path.stat().st_mtime:
            return out_path

    payload = build_terrain_payload(resolved_spec, root)
    directory.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return out_path


# ----------------------------------------------------------------------
# Start points
# ----------------------------------------------------------------------
def _unit_markers(
    source: _MapSource, layer_name: str
) -> list[tuple[int, str, tuple[float, float]]]:
    """``(team, unit name, pixel centre)`` for every team-owned unit marker."""

    markers: list[tuple[int, str, tuple[float, float]]] = []
    width = source.width
    for index, gid in enumerate(source.layer(layer_name)):
        if not gid:
            continue
        resolved = source.resolve(gid)
        if resolved is None:
            continue
        props = resolved[1]
        unit_name = props.get("unit") or props.get("customUnit")
        if not unit_name:
            continue
        team_raw = props.get("team")
        if team_raw is None:
            continue
        try:
            team = int(team_raw)
        except (TypeError, ValueError):
            # ``team="none"`` marks neutral scenery (trees on the Items layer).
            continue
        pixel = (
            (index % width) * source.tile_width + source.tile_width / 2.0,
            (index // width) * source.tile_height + source.tile_height / 2.0,
        )
        markers.append((team, unit_name, pixel))
    return markers


def start_points(
    spec: MapSpec | str, assets_root: Path | str | None = None
) -> tuple[tuple[float, float], ...]:
    """Both bases as ``(team0, team1)`` pixel centres, read from the map file.

    The command centre is the anchor when the map has one; otherwise the
    centroid of that team's markers is used (maps that only pre-place builders).
    Returns an empty tuple when either side cannot be located, so the caller can
    fall back to wherever the engine says the command centre actually is.
    :data:`MapSpec.start_points` holds the same values as constants for callers
    that must not touch the filesystem.
    """

    _resolved, source = _open_map(spec, assets_root)
    markers = _unit_markers(source, UNITS_LAYER) + _unit_markers(source, ITEMS_LAYER)
    by_team: dict[int, list[tuple[str, tuple[float, float]]]] = {}
    for team, unit_name, pixel in markers:
        by_team.setdefault(team, []).append((unit_name, pixel))

    points: list[tuple[float, float]] = []
    for team in (0, 1):
        entries = by_team.get(team) or []
        if not entries:
            return ()
        command_centres = [pixel for unit_name, pixel in entries if unit_name == "commandCenter"]
        if command_centres:
            points.append(command_centres[0])
        else:
            xs = [pixel[0] for _name, pixel in entries]
            ys = [pixel[1] for _name, pixel in entries]
            points.append((sum(xs) / len(xs), sum(ys) / len(ys)))
    return tuple(points)


# ----------------------------------------------------------------------
# CLI: regenerate every cached terrain JSON
# ----------------------------------------------------------------------
def _format_size(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.1f}"


def main(argv: Iterable[str] | None = None) -> int:
    """Regenerate ``configs/maps/*.terrain.json`` for every registered map."""

    args = list(argv if argv is not None else sys.argv[1:])
    assets_root = Path(args[0]) if args else default_assets_root()
    if not (assets_root / "maps").is_dir():
        print(f"assets root has no maps/ directory: {assets_root}", file=sys.stderr)
        return 2

    print(f"assets root: {assets_root}")
    header = f"{'map':<18}{'size(px)':>12}{'tiles':>12}{'water':>8}{'blocked':>9}  bases"
    print(header)
    failures = 0
    for spec in TWO_PLAYER_MAPS:
        try:
            path = ensure_terrain_json(spec, assets_root)
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:  # pragma: no cover - asset problem
            failures += 1
            print(f"{spec.name:<18}  FAILED: {error}", file=sys.stderr)
            continue
        blocked = sum(1 for value in payload["blocked"] if value)
        bases = start_points(spec, assets_root)
        base_text = " ".join(
            f"t{team}=({_format_size(x)},{_format_size(y)})" for team, (x, y) in enumerate(bases)
        ) or "unresolved"
        print(
            f"{spec.name:<18}{spec.width_px:>5}x{spec.height_px:<6}"
            f"{spec.width_tiles:>5}x{spec.height_tiles:<6}"
            f"{spec.water_ratio:>8.2%}{blocked / spec.tile_count:>9.2%}  {base_text}"
        )
        if len(bases) != len(spec.start_points):
            print(
                f"  WARNING: tmx start points {bases} differ in count from the "
                f"registry constants {spec.start_points}",
                file=sys.stderr,
            )
    print(f"terrain json: {TERRAIN_DIR}")
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover - manual regeneration
    raise SystemExit(main())
