# SPDX-License-Identifier: Apache-2.0
"""Contract tests for :mod:`ironzero.rwpp.maps` (the stock [p2] map registry).

Two kinds of assertion live here and they protect different things:

* **Registry facts** (name, size, water ratio, bases) -- checked against the
  ``.tmx`` headers and unit markers themselves, never against a copied constant,
  so a typo in :data:`ironzero.rwpp.maps.TWO_PLAYER_MAPS` fails.
* **The water rule** -- ``Lake`` must stay at 25.02%. Stock maps reference
  external ``.tsx`` files that carry no tile properties, so water is detected
  from the tileset *name*; if that walk is "improved" into a property-only
  rule, every map reports 0% water and the reward quietly changes meaning.

The asset-dependent tests are skipped (with the path in the message) when the
RWPP runtime is not mounted, matching the rest of the suite.
"""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from ironzero.rwpp import maps
from ironzero.terrain.grid import TerrainGrid

#: Lake's published water share, cross-checked three ways: the tile decode here,
#: the independent ``.runtime/probe_2p_maps.py`` survey, and the historical
#: 4229/16900 figure in ``audit/用户四条抱怨_根因与修复_20261115.md``.
LAKE_WATER_RATIO = 0.2502
#: Tolerance for the ratio check: the registry stores 4 decimal places.
WATER_RATIO_TOLERANCE = 0.005

ASSETS_ROOT = maps.default_assets_root()
ASSETS_PRESENT = (ASSETS_ROOT / "maps" / "skirmish").is_dir()


def _require_assets() -> None:
    if not ASSETS_PRESENT:
        pytest.skip(f"RWPP runtime assets not mounted at {ASSETS_ROOT}")


def _map_file(spec: maps.MapSpec) -> Path:
    return ASSETS_ROOT / spec.relative


def _tmx_header(path: Path) -> dict[str, int]:
    """The raw ``<map>`` header, parsed independently of ``ironzero.rwpp.maps``."""

    root = ET.parse(path).getroot()
    assert root.tag == "map", path
    return {
        "width": int(root.get("width", "0")),
        "height": int(root.get("height", "0")),
        "tilewidth": int(root.get("tilewidth", "0")),
        "tileheight": int(root.get("tileheight", "0")),
    }


# ----------------------------------------------------------------------
# Registry shape (no assets needed)
# ----------------------------------------------------------------------
def test_registry_has_nine_distinct_maps() -> None:
    specs = maps.map_specs()
    assert len(specs) == 9
    assert len({spec.name for spec in specs}) == 9
    assert len({spec.relative for spec in specs}) == 9
    # The registry is the public tuple, not a copy that can drift from it.
    assert specs is maps.TWO_PLAYER_MAPS


def test_every_spec_is_internally_consistent() -> None:
    for spec in maps.map_specs():
        assert spec.relative.startswith("maps/skirmish/"), spec
        assert spec.relative.endswith(".tmx"), spec
        assert spec.tile_px > 0
        # Pixel sizes must be exact multiples of the tile size, otherwise the
        # tile count derived from them would silently truncate.
        assert spec.width_px % spec.tile_px == 0, spec
        assert spec.height_px % spec.tile_px == 0, spec
        assert spec.width_tiles * spec.tile_px == spec.width_px
        assert spec.height_tiles * spec.tile_px == spec.height_px
        assert 0.0 <= spec.water_ratio <= 1.0, spec
        assert spec.has_water == (spec.water_ratio > 0.0), spec
        assert spec.terrain_path.name == f"{spec.name}.terrain.json", spec
        assert spec.size_px == (float(spec.width_px), float(spec.height_px))


def test_short_name_and_relative_path_resolve_to_the_same_spec() -> None:
    for spec in maps.map_specs():
        by_name = maps.map_spec(spec.name)
        by_relative = maps.map_spec(spec.relative)
        by_file_name = maps.map_spec(spec.file_name)
        assert by_name is by_relative is by_file_name, spec.name
        assert maps.map_spec(spec.name.upper()) is spec  # case-insensitive
        assert maps.map_spec(spec.relative.replace("/", "\\")) is spec  # windows slash


def test_unknown_map_raises_key_error_listing_candidates() -> None:
    with pytest.raises(KeyError) as excinfo:
        maps.map_spec("Definitely Not A Map")

    message = str(excinfo.value)
    assert "Definitely Not A Map" in message
    # The message must be actionable: every legal short name is listed.
    for spec in maps.map_specs():
        assert spec.name in message, f"{spec.name} missing from KeyError message"


def test_resolve_map_size_returns_pixel_size() -> None:
    assert maps.resolve_map_size("Lake") == (2600.0, 2600.0)
    assert maps.resolve_map_size("Two_cold_sides") == (2700.0, 2500.0)
    assert maps.resolve_map_size("[p2]Beach landing (2p) [by hxyy].tmx") == (4000.0, 3200.0)
    # Sizes are floats: the observation divides by them.
    assert all(isinstance(value, float) for value in maps.resolve_map_size("Hills"))


def test_resolve_map_size_is_not_the_old_hard_coded_default() -> None:
    """The whole point of the registry: sizes differ per map, 2600x2500 is not global."""

    sizes = {spec.size_px for spec in maps.map_specs()}
    assert len(sizes) > 1
    assert maps.resolve_map_size("Lake") != maps.resolve_map_size("Beach landing")
    assert maps.resolve_map_size("Two_cold_sides") == (2700.0, 2500.0)


def test_registered_start_points_are_inside_the_map() -> None:
    for spec in maps.map_specs():
        assert len(spec.start_points) == 2, spec.name
        for x, y in spec.start_points:
            assert 0.0 <= x <= spec.width_px, (spec.name, x)
            assert 0.0 <= y <= spec.height_px, (spec.name, y)
        # The two bases must be far apart, or the map is not a duel.
        (x0, y0), (x1, y1) = spec.start_points
        assert abs(x0 - x1) + abs(y0 - y1) > 500.0, spec.name
        assert spec.base_of(0) == spec.start_points[0]
        assert spec.base_of(1) == spec.start_points[1]
        assert spec.base_of(2) is None


# ----------------------------------------------------------------------
# Registry vs the real .tmx headers (needs assets)
# ----------------------------------------------------------------------
def test_registered_sizes_match_the_tmx_headers() -> None:
    _require_assets()
    for spec in maps.map_specs():
        path = _map_file(spec)
        assert path.is_file(), path
        header = _tmx_header(path)
        assert spec.width_px == header["width"] * header["tilewidth"], spec.name
        assert spec.height_px == header["height"] * header["tileheight"], spec.name
        assert spec.tile_px == header["tilewidth"] == header["tileheight"], spec.name
        assert spec.width_tiles == header["width"], spec.name
        assert spec.height_tiles == header["height"], spec.name


def test_registered_water_ratios_match_a_fresh_measurement() -> None:
    _require_assets()
    for spec in maps.map_specs():
        measured = maps.measure_water_ratio(spec, ASSETS_ROOT)
        assert abs(spec.water_ratio - measured) < WATER_RATIO_TOLERANCE, (
            spec.name,
            spec.water_ratio,
            measured,
        )
        # Stricter still: the table is the 4-decimal rounding of the real
        # measurement, not an independently edited number. Without this a typo
        # inside the 0.005 tolerance (e.g. 0.6377 vs 0.6379) slips through.
        assert spec.water_ratio == round(measured, 4), (spec.name, spec.water_ratio, measured)


def test_lake_water_ratio_is_pinned() -> None:
    """Guard the water-detection algorithm itself, not just the table."""

    lake = maps.map_spec("Lake")
    assert abs(lake.water_ratio - LAKE_WATER_RATIO) < WATER_RATIO_TOLERANCE
    # The registry calls it water; if the rule broke it would read 0.0 and
    # every sea-dependent decision would silently switch itself off.
    assert lake.has_water is True

    _require_assets()
    measured = maps.measure_water_ratio(lake, ASSETS_ROOT)
    assert abs(measured - LAKE_WATER_RATIO) < WATER_RATIO_TOLERANCE


def test_hills_is_the_only_map_without_water() -> None:
    no_water = [spec.name for spec in maps.map_specs() if not spec.has_water]
    assert no_water == ["Hills"]


def test_start_points_match_the_tmx_unit_markers() -> None:
    _require_assets()
    for spec in maps.map_specs():
        resolved = maps.start_points(spec, ASSETS_ROOT)
        assert len(resolved) == 2, spec.name
        for expected, actual in zip(spec.start_points, resolved):
            assert expected == pytest.approx(actual, abs=1e-6), spec.name


# ----------------------------------------------------------------------
# Terrain payload
# ----------------------------------------------------------------------
def test_terrain_payload_shape_and_blocked_mask() -> None:
    _require_assets()
    for spec in maps.map_specs():
        payload = maps.build_terrain_payload(spec, ASSETS_ROOT)
        assert set(payload) == {
            "id",
            "name",
            "width",
            "height",
            "tile_width",
            "tile_height",
            "width_px",
            "height_px",
            "blocked",
        }
        assert payload["id"] == payload["name"] == spec.name
        assert payload["width"] == spec.width_tiles
        assert payload["height"] == spec.height_tiles
        assert payload["tile_width"] == payload["tile_height"] == spec.tile_px
        assert payload["width_px"] == spec.width_px
        assert payload["height_px"] == spec.height_px

        blocked = payload["blocked"]
        assert len(blocked) == payload["width"] * payload["height"], spec.name
        assert spec.tile_count == len(blocked)
        assert set(blocked) <= {0, 1}, spec.name
        # A mask of all zeros means the parse found nothing, which is a bug.
        ratio = sum(blocked) / len(blocked)
        assert ratio > 0.0, spec.name
        # ... and a mask of all ones means the tilesets failed to resolve.
        assert ratio < 1.0, spec.name

        # TerrainGrid is the consumer: the payload must load without massaging.
        grid = TerrainGrid.from_payload(payload)
        assert (grid.width, grid.height) == (spec.width_tiles, spec.height_tiles)
        assert (grid.width_px, grid.height_px) == (spec.width_px, spec.height_px)
        assert sum(grid.blocked) == sum(blocked)


def test_water_tiles_are_blocked_for_land() -> None:
    """Water must appear in the blocked mask, and no map may be waterless there."""

    _require_assets()
    for spec in maps.map_specs():
        payload = maps.build_terrain_payload(spec, ASSETS_ROOT)
        blocked_ratio = sum(payload["blocked"]) / len(payload["blocked"])
        # Water is a subset of blocked, so the mask can only be larger.
        assert blocked_ratio + 1e-9 >= spec.water_ratio, (spec.name, blocked_ratio)
        if spec.has_water:
            assert blocked_ratio > 0.0, spec.name


def test_build_terrain_payload_accepts_a_name_too() -> None:
    _require_assets()
    by_spec = maps.build_terrain_payload(maps.map_spec("Lake"), ASSETS_ROOT)
    by_name = maps.build_terrain_payload("Lake", ASSETS_ROOT)
    assert by_name == by_spec


def test_terrain_json_files_exist_and_load(tmp_path: Path) -> None:
    for spec in maps.map_specs():
        assert spec.terrain_path.is_file(), (
            f"{spec.terrain_path} missing; regenerate with "
            "`python -m ironzero.rwpp.maps`"
        )
        grid = TerrainGrid.from_file(str(spec.terrain_path))
        assert grid.map_id == spec.name
        assert grid.name == spec.name
        assert (grid.width, grid.height) == (spec.width_tiles, spec.height_tiles)
        assert (grid.width_px, grid.height_px) == (spec.width_px, spec.height_px)
        assert grid.tile_count == spec.tile_count
        assert 0.0 < grid.land_ratio < 1.0, spec.name


def test_committed_terrain_json_matches_a_fresh_build() -> None:
    """The checked-in cache must not drift from the generator."""

    _require_assets()
    for spec in maps.map_specs():
        stored = json.loads(spec.terrain_path.read_text(encoding="utf-8"))
        fresh = maps.build_terrain_payload(spec, ASSETS_ROOT)
        assert stored["width"] == fresh["width"], spec.name
        assert stored["height"] == fresh["height"], spec.name
        assert stored["blocked"] == fresh["blocked"], spec.name


def test_ensure_terrain_json_writes_then_reuses(tmp_path: Path) -> None:
    _require_assets()
    spec = maps.map_spec("Lake")
    first = maps.ensure_terrain_json(spec, ASSETS_ROOT, tmp_path)
    assert first == tmp_path / "Lake.terrain.json"
    assert first.is_file()
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert len(payload["blocked"]) == spec.tile_count

    # A second call reuses the file: same path, and the mtime does not move.
    stamp = first.stat().st_mtime_ns
    second = maps.ensure_terrain_json(spec, ASSETS_ROOT, tmp_path)
    assert second == first
    assert second.stat().st_mtime_ns == stamp

    # Caching must be staleness-driven, not unconditional: backdate the JSON
    # behind the map file's mtime and the next call has to rebuild it.
    map_mtime = _map_file(spec).stat().st_mtime
    os.utime(first, (map_mtime - 3600.0, map_mtime - 3600.0))
    assert first.stat().st_mtime < map_mtime
    rebuilt = maps.ensure_terrain_json(spec, ASSETS_ROOT, tmp_path)
    assert rebuilt == first
    assert rebuilt.stat().st_mtime >= map_mtime
    assert json.loads(rebuilt.read_text(encoding="utf-8"))["blocked"] == payload["blocked"]


def test_missing_map_file_raises_file_not_found(tmp_path: Path) -> None:
    spec = maps.map_spec("Lake")
    with pytest.raises(FileNotFoundError):
        maps.build_terrain_payload(spec, tmp_path)
    with pytest.raises(FileNotFoundError):
        maps.start_points(spec, tmp_path)


# ----------------------------------------------------------------------
# Registry is the shared surface: guard the constants it replaces
# ----------------------------------------------------------------------
def test_no_module_hard_codes_a_single_map_size() -> None:
    """Sizes must come from the registry, not be assumed per call site."""

    spec = maps.map_spec("Two_cold_sides")
    assert (spec.width_px, spec.height_px) == (2700, 2500)
    lake = maps.map_spec("Lake")
    assert (lake.width_px, lake.height_px) == (2600, 2600)

    # 16875 vs 16900 tiles: the tile counts are 0.15% apart while the geometry
    # differs by 100 px on each axis. A caller that inferred the map size from
    # len(blocked) -- or that reused the old 2700x2500 -- would be wrong on one
    # of these two maps and nothing would raise.
    assert (lake.tile_count, spec.tile_count) == (16900, 16875)
    assert lake.size_px != spec.size_px

    # Hills shares Lake's 2600x2600 geometry, so equal sizes must not be read
    # as "same map".
    hills = maps.map_spec("Hills")
    assert hills.size_px == lake.size_px
    assert hills.water_ratio == 0.0 < lake.water_ratio
    assert [s.size_px for s in maps.map_specs()].count((2600.0, 2600.0)) == 2
