# SPDX-License-Identifier: Apache-2.0
"""Tests for the sandbox ``unitSprites`` manifest (``docs/沙盘贴图契约.md`` §1).

These tests run against the real ``.game_cache`` when it is present (it is
git-ignored and prepared by ``web/scripts/prepare_assets.ps1``). When the cache
is absent every cache-dependent test skips with an actionable message instead of
failing, so a fresh clone still gets a green ``pytest -q``.

The frame-count tests deliberately re-derive the frame layout from the sprite's
own alpha channel instead of trusting ``total_frames``: a sheet whose pixels do
not repeat with the declared frame width would render as garbage even though
``frames * frameW == image.width`` happens to hold.
"""

from __future__ import annotations

import inspect
import struct
import sys
from pathlib import Path

import pytest

from web.backend import maps as gamemaps
from web.backend import unit_sprites as gamesprites

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = REPO_ROOT / ".game_cache"

#: The 31 distinct ``unit_type`` values that appear across the recorded real
#: replays under ``runs/platform/**/replays`` (32 with the generic ``building``
#: pseudo-type). Grown from 28 in the reference replay
#: ``real-rwpp-20261111-092721-4f66bf`` by scanning every replay on disk.
REAL_UNIT_TYPES = (
    "airFactory",
    "amphibiousJet",
    "battleShip",
    "builder",
    "building",
    "c_antiAirTurret",
    "c_antiAirTurretT2",
    "c_artillery",
    "c_helicopter",
    "c_interceptor",
    "c_tank",
    "c_turret_t1",
    "c_turret_t1_artillery",
    "c_turret_t2_gun",
    "commandCenter",
    "dropship",
    "extractorT1",
    "extractorT2",
    "fabricatorT1",
    "factory",
    "gunBoat",
    "gunShip",
    "hoverTank",
    "hovercraft",
    "landFactory",
    "lightGunship",
    "mechFactory",
    "missileShip",
    "scout",
    "seaFactory",
    "tank",
)

requires_cache = pytest.mark.skipif(
    not (CACHE_ROOT / "units" / "game").is_dir()
    or not (CACHE_ROOT / "units" / "drawable").is_dir(),
    reason="game asset cache absent - run web/scripts/prepare_assets.ps1",
)


def _png_size(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    assert header[:8] == b"\x89PNG\r\n\x1a\n", path
    width, height = struct.unpack(">II", header[16:24])
    return width, height


@pytest.fixture(scope="module")
def manifest() -> dict:
    return gamesprites.build_manifest()


def _body_and_turret(label: str, entry: dict):
    yield label, entry
    if entry.get("turretImage"):
        yield f"{label}.turret", entry


# ----------------------------------------------------------------------
# Structure
# ----------------------------------------------------------------------
@requires_cache
def test_manifest_top_level_shape(manifest: dict) -> None:
    assert set(manifest) == {"version", "byType", "aliases", "fallback"}
    assert manifest["version"] == gamesprites.MANIFEST_VERSION
    assert isinstance(manifest["byType"], dict) and manifest["byType"]
    assert isinstance(manifest["aliases"], dict)


@requires_cache
def test_fallback_has_all_three_keys(manifest: dict) -> None:
    assert set(manifest["fallback"]) == {"mobile", "builder", "building"}
    for name, entry in manifest["fallback"].items():
        for field in ("image", "frames", "frameW", "frameH", "scale", "kind"):
            assert field in entry, f"fallback.{name} missing {field}"
    assert manifest["fallback"]["mobile"]["kind"] == "mobile"
    assert manifest["fallback"]["builder"]["kind"] == "mobile"
    assert manifest["fallback"]["building"]["kind"] == "building"


@requires_cache
def test_every_entry_has_required_fields(manifest: dict) -> None:
    for label, entry in gamesprites.iter_manifest_entries(manifest):
        for field in ("image", "frames", "frameW", "frameH", "scale", "kind"):
            assert field in entry, f"{label} missing {field}"
        assert entry["kind"] in {"building", "mobile"}, label
        assert entry["frames"] >= 1, label
        assert entry["frameW"] >= 1 and entry["frameH"] >= 1, label
        assert entry["scale"] > 0, label


# ----------------------------------------------------------------------
# Existence + geometry self-consistency
# ----------------------------------------------------------------------
@requires_cache
def test_every_image_exists(manifest: dict) -> None:
    missing = []
    for label, entry in gamesprites.iter_manifest_entries(manifest):
        path = CACHE_ROOT / entry["image"]
        if not path.is_file():
            missing.append((label, entry["image"]))
    assert not missing, f"manifest references missing images: {missing}"


@requires_cache
def test_frames_times_framew_equals_image_width(manifest: dict) -> None:
    """Contract §1.2: ``frames * frameW == image.width``."""

    problems = []
    for label, entry in gamesprites.iter_manifest_entries(manifest):
        width, _height = _png_size(CACHE_ROOT / entry["image"])
        expected = entry["frames"] * entry["frameW"]
        if expected != width:
            problems.append((label, entry["image"], width, expected))
    assert not problems, f"frames*frameW != image width: {problems}"


@requires_cache
def test_frameh_equals_image_height(manifest: dict) -> None:
    """Contract §1.2: ``frameH == image.height`` (row-major slicing)."""

    problems = []
    for label, entry in gamesprites.iter_manifest_entries(manifest):
        _width, height = _png_size(CACHE_ROOT / entry["image"])
        if entry["frameH"] != height:
            problems.append((label, entry["image"], height, entry["frameH"]))
    assert not problems, f"frameH != image height: {problems}"


@requires_cache
def test_no_vertical_strips_today(manifest: dict) -> None:
    """Every shipped sheet is cut on X only; ``layout`` marks the exception."""

    offenders = [
        (label, entry["image"])
        for label, entry in gamesprites.iter_manifest_entries(manifest)
        if entry.get("layout", "horizontal") != "horizontal"
    ]
    assert not offenders, f"unexpected vertical-strip layout: {offenders}"


@requires_cache
def test_scale_values_are_sane(manifest: dict) -> None:
    for label, entry in gamesprites.iter_manifest_entries(manifest):
        assert 0.05 <= entry["scale"] <= 8.0, f"{label} scale {entry['scale']}"


# ----------------------------------------------------------------------
# Pixel-level frame verification (independent of the ini's total_frames)
# ----------------------------------------------------------------------
def _alpha_period(path: Path) -> int | None:
    """Smallest p where column x matches column x+p for all rows, else ``None``.

    Frames in these sheets sit side by side with identical alpha silhouettes,
    so ``width / p`` is the frame count the engine actually slices at.
    """

    Image = pytest.importorskip("PIL.Image")
    image = Image.open(path).convert("RGBA")
    width, height = image.size
    pixels = image.getchannel("A").load()
    for period in range(1, width // 2 + 1):
        if width % period:
            continue
        if all(
            pixels[x, y] == pixels[x + period, y]
            for x in range(width - period)
            for y in range(height)
        ):
            return period
    return None


@requires_cache
def test_multi_frame_sheets_match_their_pixel_period(manifest: dict) -> None:
    """A declared frame count must agree with the sprite's own repetition.

    Catches the classic failure: ``total_frames: 4`` on a sheet that only has 3
    frames (``frames * frameW == width`` can still hold for the wrong divisor).
    Sheets whose alpha does not repeat at all (organic bug sprites, single-frame
    art with soft edges) are skipped — they cannot be verified this way.
    """

    mismatches = []
    checked = 0
    for label, entry in gamesprites.iter_manifest_entries(manifest):
        if entry["frames"] <= 1:
            continue
        period = _alpha_period(CACHE_ROOT / entry["image"])
        if period is None:
            continue
        checked += 1
        implied = _png_size(CACHE_ROOT / entry["image"])[0] // period
        if implied != entry["frames"]:
            mismatches.append((label, entry["image"], entry["frames"], implied))
    assert checked >= 10, f"only {checked} multi-frame sheets verifiable - check the cache"
    assert not mismatches, f"declared frames disagree with pixels: {mismatches}"


# ----------------------------------------------------------------------
# Real replay coverage
# ----------------------------------------------------------------------
@requires_cache
def test_real_unit_types_resolve(manifest: dict) -> None:
    """Every real ``unit_type`` resolves to a sprite, directly or via alias.

    ``building`` is the one deliberate exception: it is a generic pseudo-type
    written by the legacy recorders (``scripts/record_rwpp_replay.py``) when the
    bridge reports no concrete ``unit_type``. There is no such unit in the game,
    so it is *supposed* to reach ``fallback["building"]`` via the replay's own
    ``is_building`` flag (contract §2.2), not ``byType``.
    """

    generic = {"building"}
    unresolved = [
        t
        for t in REAL_UNIT_TYPES
        if t not in generic and gamesprites.resolve_type(manifest, t) is None
    ]
    assert not unresolved, f"unit types with no sprite at all: {unresolved}"

    # The generic pseudo-types must not resolve either, so they reach fallback.
    for pseudo in generic:
        assert gamesprites.resolve_type(manifest, pseudo) is None
        assert manifest["fallback"][pseudo]["kind"] == "building"


@requires_cache
def test_bytype_hit_count_at_least_25(manifest: dict) -> None:
    """Contract acceptance: at least 25 of the real types hit ``byType`` itself.

    The remainder are allowed to fall through to ``fallback`` but must still be
    few: everything except the generic ``building`` pseudo-type is expected to
    have real art.
    """

    by_type = manifest["byType"]
    direct = [t for t in REAL_UNIT_TYPES if t in by_type]
    assert len(direct) >= 25, f"only {len(direct)} direct hits: {sorted(set(REAL_UNIT_TYPES) - set(direct))}"

    via_alias = [t for t in REAL_UNIT_TYPES if t not in by_type and gamesprites.resolve_type(manifest, t)]
    unresolved = set(REAL_UNIT_TYPES) - set(direct) - set(via_alias)
    assert unresolved <= {"building"}, f"unexpected fallback types: {sorted(unresolved)}"


@requires_cache
def test_legacy_classic_types_have_real_art(manifest: dict) -> None:
    """The builtin Android-era types must not fall back to a generic tank."""

    for name in gamesprites.CLASSIC_SPRITES:
        assert name in manifest["byType"], f"classic type {name} missing from byType"
        entry = manifest["byType"][name]
        assert "drawable" in entry["image"], f"{name} should use builtin drawable art"


@requires_cache
def test_specific_units_get_correct_sizes(manifest: dict) -> None:
    """Spot-check the sizes called out in the contract's rationale (§0)."""

    by_type = manifest["byType"]

    # command center 53x68 (4 frames), tank 16x30 (3 frames) - must differ.
    cc = by_type["commandCenter"]
    assert (cc["frameW"], cc["frameH"], cc["frames"]) == (53, 68, 4)
    assert cc["kind"] == "building"

    tank = by_type["c_tank"]
    assert (tank["frameW"], tank["frameH"], tank["frames"]) == (16, 30, 3)
    assert tank["scale"] == pytest.approx(0.8)
    assert tank["turretImage"].endswith("tank_turret.png")

    extractor = by_type["extractorT1"]
    assert (extractor["frameW"], extractor["frames"]) == (37, 4)
    assert extractor["kind"] == "building"


@requires_cache
def test_aliases_all_resolve(manifest: dict) -> None:
    by_type = manifest["byType"]
    for alias, target in manifest["aliases"].items():
        assert target in by_type, f"alias {alias} -> missing {target}"
        assert alias not in by_type, f"alias {alias} shadows a real unit type"


@requires_cache
def test_building_kind_matches_replay_flags(manifest: dict) -> None:
    """``kind == building`` must agree with the replay's own ``is_building``."""

    buildings = {
        "airFactory",
        "c_antiAirTurret",
        "c_antiAirTurretT2",
        "c_turret_t1",
        "c_turret_t1_artillery",
        "c_turret_t2_gun",
        "commandCenter",
        "extractorT1",
        "extractorT2",
        "fabricatorT1",
        "landFactory",
        "mechFactory",
        "seaFactory",
    }
    by_type = manifest["byType"]
    for name in buildings:
        entry = gamesprites.resolve_type(manifest, name)
        assert entry is not None, name
        assert entry["kind"] == "building", f"{name} should be a building"

    mobiles = {"builder", "c_tank", "hoverTank", "scout", "dropship", "gunBoat", "battleShip"}
    for name in mobiles:
        entry = gamesprites.resolve_type(manifest, name)
        assert entry is not None, name
        assert entry["kind"] == "mobile", f"{name} should be mobile"


# ----------------------------------------------------------------------
# Failure modes: the manifest must never degrade silently
# ----------------------------------------------------------------------
def test_missing_image_raises(tmp_path: Path) -> None:
    """A manifest pointing at absent art must raise, not return a bad entry."""

    with pytest.raises(gamesprites.UnitSpriteError):
        gamesprites._sprite_entry(
            tmp_path, "units/drawable/nope.png", 1, 1.0, "mobile", "test"
        )


def test_bad_frame_count_raises(tmp_path: Path) -> None:
    """Width not divisible by the frame count must raise (not silently shift)."""

    Image = pytest.importorskip("PIL.Image")
    png = tmp_path / "sheet.png"
    Image.new("RGBA", (35, 20), (0, 0, 0, 255)).save(png)
    with pytest.raises(gamesprites.UnitSpriteError):
        gamesprites._sprite_entry(tmp_path, "sheet.png", 4, 1.0, "mobile", "test")


def test_vertical_layout_is_opt_in(tmp_path: Path) -> None:
    """Contract §1.2's vertical-strip exception must be requested explicitly."""

    Image = pytest.importorskip("PIL.Image")
    png = tmp_path / "strip.png"
    # 15 divides by neither 5 nor 3 into a clean strip on X; 40 divides by 4 on Y,
    # so only the vertical axis can produce 4 frames.
    Image.new("RGBA", (15, 40), (0, 0, 0, 255)).save(png)

    # Horizontal slicing is the default, so 4 frames on a 15x40 sheet is an error.
    with pytest.raises(gamesprites.UnitSpriteError):
        gamesprites._sprite_entry(tmp_path, "strip.png", 4, 1.0, "mobile", "test")

    # Opting in switches the axis and records the layout in the manifest.
    entry = gamesprites._sprite_entry(
        tmp_path, "strip.png", 4, 1.0, "mobile", "test", layout="vertical"
    )
    assert entry["layout"] == "vertical"
    assert (entry["frameW"], entry["frameH"]) == (15, 10)


def test_missing_cache_root_raises(tmp_path: Path) -> None:
    with pytest.raises(gamesprites.UnitSpriteError):
        gamesprites.build_manifest(tmp_path)


def test_unit_sprites_manifest_degrades_to_empty(tmp_path: Path) -> None:
    """The route helper returns a valid empty manifest instead of raising."""

    empty = gamesprites.unit_sprites_manifest(tmp_path)
    assert empty["version"] == gamesprites.MANIFEST_VERSION
    assert empty["byType"] == {}
    assert empty["fallback"] == {}


# ----------------------------------------------------------------------
# Wiring: /api/maps must expose unitSprites and keep units
# ----------------------------------------------------------------------
def test_api_maps_exposes_unitsprites() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from web.backend.app import create_app

    with TestClient(create_app()) as client:
        response = client.get("/api/maps")
    assert response.status_code == 200
    payload = response.json()
    assert "unitSprites" in payload, "contract §1 requires the unitSprites field"
    assert "units" in payload, "legacy units field must stay for old frontends"
    if not payload["available"]:
        pytest.skip("game asset cache not mounted in this environment")
    sprites = payload["unitSprites"]
    assert sprites["version"] == gamesprites.MANIFEST_VERSION
    assert sprites["byType"], "unitSprites.byType must not be empty with assets present"


def test_route_source_mentions_unit_sprites() -> None:
    """Cheap guard so a refactor cannot silently drop the field again."""

    from web.backend.app.routers import maps as maps_router

    source = inspect.getsource(maps_router)
    assert "unitSprites" in source
    assert "unit_sprites_manifest" in source
