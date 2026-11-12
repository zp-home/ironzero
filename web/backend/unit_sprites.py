# SPDX-License-Identifier: Apache-2.0
"""Build the sandbox ``unitSprites`` manifest from the local game cache.

This module is the backend half of ``docs/沙盘贴图契约.md`` §1. It turns the
copied Rusted Warfare assets (``.game_cache/units/game`` + ``.game_cache/units/drawable``,
see ``web/scripts/prepare_assets.ps1``) into a manifest the Web sandbox can
render with 1:1 world pixels, correct frame slicing and correct body/turret
rotation.

Why it exists: ``maps.py:unit_sprites()`` used to expose 8 hard-coded keys, so a
real replay with 28 distinct ``unit_type`` values rendered almost every unit
with the vector fallback. Real replays contain two naming families:

* **mod / 1.15 ini units** — every ``.ini`` under ``assets/units`` declares
  ``[core] name:`` (``c_tank``, ``c_turret_t1``, ``extractorT1`` …). These are
  authoritative: geometry comes from the ini itself.
* **legacy builtin units** — the Android-era built-in types that ship inside
  ``game-lib.jar`` (``commandCenter``, ``landFactory``, ``builder`` …). They
  have **no** ini and their art lives in ``res/drawable``. They are covered by
  the explicit :data:`CLASSIC_SPRITES` table below, whose evidence column
  records how each row was derived.

The manifest never degrades silently: a missing image or an image whose size
disagrees with ``total_frames`` raises :class:`UnitSpriteError`.
"""

from __future__ import annotations

import re
import struct
from pathlib import Path
from typing import Any, Iterable, Mapping

__all__ = [
    "MANIFEST_VERSION",
    "SPRITE_MAX_FRAMES",
    "UnitSpriteError",
    "build_manifest",
    "unit_sprites_manifest",
]

#: Bumped only when the manifest shape changes in a way the frontend must notice.
MANIFEST_VERSION = 1

#: Sanity ceiling: no real unit sheet has more frames than this on one row.
SPRITE_MAX_FRAMES = 64

#: Where the copied assets live, relative to ``ASSETS_ROOT``.
GAME_UNITS_DIR = "units/game"
DRAWABLE_DIR = "units/drawable"
SHARED_IMAGE_PREFIX = "SHARED:"

#: ``image: NONE`` means "no image" in the game's own ini dialect.
_NO_IMAGE = "NONE"


class UnitSpriteError(RuntimeError):
    """Raised when the sprite manifest cannot be produced faithfully."""


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def _png_size(path: Path) -> tuple[int, int]:
    """Read width/height straight out of the PNG IHDR chunk."""

    with path.open("rb") as handle:
        header = handle.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise UnitSpriteError(f"not a PNG: {path}")
    width, height = struct.unpack(">II", header[16:24])
    if width <= 0 or height <= 0:
        raise UnitSpriteError(f"PNG has empty size: {path}")
    return width, height


def _assets_root() -> Path:
    """Resolve the asset root exactly like :mod:`web.backend.maps` does.

    Read through the ``maps`` module at call time (not import time) so that
    tests which monkeypatch ``maps.ASSETS_ROOT`` stay authoritative.
    """

    from . import maps as gamemaps

    return Path(gamemaps.ASSETS_ROOT)


def _as_int(raw: str | None, what: str, where: str) -> int:
    if raw is None or raw == "":
        raise UnitSpriteError(f"{where}: missing {what}")
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise UnitSpriteError(f"{where}: {what} is not an integer: {raw!r}") from exc


def _as_scale(raw: str | None, default: float = 1.0) -> float:
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw.strip())
    except ValueError:
        return default
    return value if value > 0 else default


def _as_flag(raw: str | None) -> bool:
    if raw is None:
        return False
    return raw.strip().lower() in {"true", "1", "yes"}


# ----------------------------------------------------------------------
# ini parsing
# ----------------------------------------------------------------------
def parse_ini(path: Path) -> dict[str, dict[str, str]]:
    """Parse one game ini into ``{section_lower: {key: first_value}}``.

    The dialect is close to but not identical to INI:

    * ``#`` starts a comment (also used to disable entries);
    * keys are ``key: value`` or ``key:value``;
    * section names are case-insensitive and appear as ``[core]``;
    * the *first* occurrence wins — later duplicates are the game's own
      "override" habit and are not what the renderer needs.
    """

    sections: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        header = re.match(r"^\[(.+)\]$", stripped)
        if header:
            name = header.group(1).strip().lower()
            current = sections.setdefault(name, {})
            continue
        if current is None or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        if not key or key in current:
            continue
        current[key] = value.strip()
    return sections


class IniUnit:
    """One ``.ini`` with ``copyFrom`` inheritance already resolved.

    ``origins`` records, per ``section/key``, the directory of the ini that
    actually declared the value. That matters for images: a key inherited from
    a ``copyFrom`` parent must be resolved against the *parent's* folder, not
    the child's.
    """

    __slots__ = ("path", "sections", "origins", "name", "chains")

    def __init__(
        self,
        path: Path,
        sections: dict[str, dict[str, str]],
        chains: tuple[str, ...],
        origins: dict[tuple[str, str], Path] | None = None,
    ):
        self.path = path
        self.sections = sections
        self.chains = chains
        self.origins = origins or {}
        self.name = sections.get("core", {}).get("name", "").strip()

    def origin_dir(self, section: str, key: str) -> Path:
        return self.origins.get((section, key), self.path.parent)

    def _get(self, section: str, key: str) -> str | None:
        value = self.sections.get(section, {}).get(key)
        if value is None or value == "":
            return None
        return value

    @property
    def is_building(self) -> bool:
        return _as_flag(self._get("core", "isBuilding"))

    @property
    def is_builder(self) -> bool:
        return _as_flag(self._get("core", "isBuilder"))

    @property
    def image(self) -> str | None:
        return self._clean_image(self._get("graphics", "image"))

    @property
    def turret_image(self) -> str | None:
        return self._clean_image(self._get("graphics", "image_turret"))

    @staticmethod
    def _clean_image(value: str | None) -> str | None:
        if value is None:
            return None
        if value.strip().upper() == _NO_IMAGE:
            return None
        return value.strip()

    @property
    def total_frames(self) -> int:
        raw = self._get("graphics", "total_frames")
        if raw is None:
            return 1
        frames = _as_int(raw, "total_frames", str(self.path))
        return max(1, frames)

    @property
    def turret_total_frames(self) -> int:
        raw = self._get("graphics", "turret_total_frames")
        if raw is None:
            return 1
        return max(1, _as_int(raw, "turret_total_frames", str(self.path)))

    @property
    def image_scale(self) -> float:
        return _as_scale(self._get("graphics", "imageScale"))

    @property
    def turret_image_scale(self) -> float:
        # The engine falls back to the body scale when turretImageScale is
        # absent (e.g. turret_t1.ini sets neither, tank.ini sets both to 0.8).
        return _as_scale(self._get("graphics", "turretImageScale"), self.image_scale)


def _index_inis(game_root: Path) -> dict[str, list[Path]]:
    """Index every ini by file name and by stem so ``copyFrom`` can resolve.

    ``copyFrom`` names a *file* (``turret_common_land.ini``), sometimes with the
    extension (``extractor_common.ini``) and sometimes without. Stem collisions
    are real — ``bug_base/bug_generator/bug_generator.ini`` and
    ``classic_bugs/bug_generator/bug_generator.ini`` both exist — so every
    candidate is kept and the nearest one wins (see :func:`_pick_parent`).

    ``copyFrom: 1`` / ``2`` … are leg- and turret-local copies, not files.
    """

    index: dict[str, list[Path]] = {}
    for path in sorted(game_root.rglob("*.ini")):
        index.setdefault(path.name.lower(), []).append(path)
        index.setdefault(path.stem.lower(), []).append(path)
    return index


def _pick_parent(candidates: list[Path], child: Path) -> Path:
    """Choose the copyFrom parent the game would choose: same folder first."""

    for candidate in candidates:
        if candidate.parent == child.parent:
            return candidate
    return candidates[0]


def load_ini_units(game_root: Path) -> list[IniUnit]:
    """Load every ini and flatten its ``copyFrom`` chain.

    A child overrides its parent key-by-key; sections the child omits are
    inherited wholesale (that is how ``turret_t1.ini`` gets ``isBuilding`` and
    ``image: turret_base.png`` from ``turret_common_land.ini``). Inheritance is
    depth-limited so a malformed cycle fails loudly instead of hanging.
    """

    index = _index_inis(game_root)
    resolved: dict[Path, IniUnit] = {}

    def resolve(path: Path, chain: tuple[str, ...] = ()) -> IniUnit:
        if path in resolved:
            return resolved[path]
        if len(chain) > 8:
            raise UnitSpriteError(
                "copyFrom chain too deep (cycle?): " + " -> ".join(chain + (path.name,))
            )
        own = parse_ini(path)
        merged: dict[str, dict[str, str]] = {}
        origins: dict[tuple[str, str], Path] = {}
        parent_name = own.get("core", {}).get("copyFrom", "").strip()
        inherits: tuple[str, ...] = ()
        if parent_name and not parent_name.isdigit():
            candidates = index.get(parent_name.lower()) or index.get(
                parent_name.lower() + ".ini"
            )
            if not candidates:
                raise UnitSpriteError(
                    f"{path.name}: copyFrom target not found: {parent_name!r}"
                )
            parent = resolve(_pick_parent(candidates, path), chain + (path.name,))
            for section, values in parent.sections.items():
                merged[section] = dict(values)
            origins.update(parent.origins)
            inherits = parent.chains
        for section, values in own.items():
            merged.setdefault(section, {}).update(values)
            for key in values:
                origins[(section, key)] = path.parent
        unit = IniUnit(path, merged, inherits + (path.name,), origins)
        resolved[path] = unit
        return unit

    return [resolve(path) for path in sorted(game_root.rglob("*.ini"))]


# ----------------------------------------------------------------------
# Image -> sprite entry
# ----------------------------------------------------------------------
def _sprite_entry(
    root: Path,
    relative_image: str,
    frames: int,
    scale: float,
    kind: str,
    what: str,
    layout: str = "horizontal",
) -> dict[str, Any]:
    """Build one manifest entry, verifying it against the real PNG size.

    ``layout`` is explicit on purpose. The default (``"horizontal"``, frames
    side by side) is what every shipped sheet uses. ``"vertical"`` exists only
    for contract §1.2's stated exception and must be requested by the caller,
    because guessing it would happily accept a bogus frame count whenever the
    *height* happens to divide evenly.
    """

    path = root / relative_image
    if not path.is_file():
        raise UnitSpriteError(f"{what}: image does not exist: {relative_image}")

    width, height = _png_size(path)
    if frames < 1:
        raise UnitSpriteError(f"{what}: frames must be >= 1, got {frames}")
    if frames > SPRITE_MAX_FRAMES:
        raise UnitSpriteError(f"{what}: implausible frame count {frames}")

    if layout == "horizontal":
        if width % frames:
            raise UnitSpriteError(
                f"{what}: {relative_image} is {width}x{height}; width is not "
                f"divisible by frames={frames}, so it cannot be sliced on X"
            )
        frame_w, frame_h = width // frames, height
    elif layout == "vertical":
        if height % frames:
            raise UnitSpriteError(
                f"{what}: {relative_image} is {width}x{height}; height is not "
                f"divisible by frames={frames}, so it cannot be sliced on Y"
            )
        frame_w, frame_h = width, height // frames
    else:
        raise UnitSpriteError(f"{what}: unknown layout {layout!r}")

    entry: dict[str, Any] = {
        "image": relative_image,
        "frames": frames,
        "frameW": frame_w,
        "frameH": frame_h,
        "scale": scale,
        "kind": kind,
    }
    if layout != "horizontal":
        entry["layout"] = layout
    return entry


def _resolve_image_path(unit: IniUnit, game_root: Path, image: str, key: str) -> str:
    """Map a ``[graphics] image:`` value onto a path relative to ASSETS_ROOT.

    ``SHARED:foo.png`` means "the sibling ``shared/`` folder of the units tree"
    (``assets/units/shared``); everything else is relative to the folder of the
    ini that *declared* the key — inherited values resolve against the parent's
    folder, which is where the file actually lives.
    """

    if image.startswith(SHARED_IMAGE_PREFIX):
        tail = image.split(":", 1)[1]
        candidate = game_root / "shared" / tail
        if not candidate.is_file():
            raise UnitSpriteError(
                f"{unit.path.name}: shared image does not exist: {image}"
            )
        return f"{GAME_UNITS_DIR}/shared/{tail}"

    origin = unit.origin_dir("graphics", key)
    candidate = origin / image
    if not candidate.is_file():
        raise UnitSpriteError(
            f"{unit.path.name}: image does not exist: {image} "
            f"(resolved against {origin.relative_to(game_root).as_posix() or '.'})"
        )
    rel = candidate.relative_to(game_root).as_posix()
    return f"{GAME_UNITS_DIR}/{rel}"


def _entry_from_ini(unit: IniUnit, root: Path, game_root: Path) -> dict[str, Any] | None:
    image = unit.image
    if image is None:
        return None
    body = _sprite_entry(
        root,
        _resolve_image_path(unit, game_root, image, "image"),
        unit.total_frames,
        unit.image_scale,
        "building" if unit.is_building else "mobile",
        f"{unit.path.name} (unit {unit.name})",
    )

    turret = unit.turret_image
    if turret is not None:
        turret_entry = _sprite_entry(
            root,
            _resolve_image_path(unit, game_root, turret, "image_turret"),
            unit.turret_total_frames,
            unit.turret_image_scale,
            "building" if unit.is_building else "mobile",
            f"{unit.path.name} (unit {unit.name}) turret",
        )
        body["turretImage"] = turret_entry["image"]
        body["turretFrames"] = turret_entry["frames"]
        body["turretFrameW"] = turret_entry["frameW"]
        body["turretFrameH"] = turret_entry["frameH"]
        body["turretScale"] = turret_entry["scale"]
    return body


# ----------------------------------------------------------------------
# Legacy builtin ("classic") unit types
# ----------------------------------------------------------------------
# These unit types have no ini: they are compiled into ``game-lib.jar`` as the
# ``com.corrodinggames.rts.game.units.ar`` enum, and their art is a plain PNG in
# the game's ``res/drawable`` (copied to ``units/drawable``). The mapping below
# is *not* guesswork — each row's evidence is the enum constant's implementation
# class (found by decoding ``ar.<clinit>()`` and the per-constant subclass) and
# that class's ``R$drawable`` references (found in its constant pool).
#
# ``frames`` was then derived from the sprites themselves: the alpha channel of
# every sheet repeats with a period equal to the frame width (base.png repeats
# every 53 px, air_factory.png every 40 px, tank2.png every 16 px, …), which is
# exactly the row-major slicing the engine uses. ``tests/test_unit_sprites.py``
# re-derives those periods with Pillow and fails if this table drifts.
#
# ``scale`` is 1.0 and ``kind`` is "mobile" unless noted: classic art is drawn
# at native pixel size, and only buildings skip rotation.
CLASSIC_SPRITES: dict[str, dict[str, Any]] = {
    # ar$45 -> units/d/e -> base.png (212x68, 4 frames of 53x68) + base_back.png
    "commandCenter": {
        "image": f"{DRAWABLE_DIR}/base.png",
        "frames": 4,
        "scale": 1.0,
        "kind": "building",
        "evidence": "ar$45 -> units/d/e",
    },
    # ar$52 -> units/e/b -> builder.png (20x20)
    "builder": {
        "image": f"{DRAWABLE_DIR}/builder.png",
        "frames": 1,
        "scale": 1.0,
        "kind": "mobile",
        "evidence": "ar$52 -> units/e/b",
    },
    # ar$12 -> units/d/m -> land_factory*.png. The engine composites the back
    # layer plus the front door layer; land_factory.png is the same building as
    # a single self-contained sprite, which is what a top-down sandbox needs
    # (layering the two halves would double-draw the footprint).
    "landFactory": {
        "image": f"{DRAWABLE_DIR}/land_factory.png",
        "frames": 1,
        "scale": 1.0,
        "kind": "building",
        "evidence": "ar$12 -> units/d/m",
    },
    # ar$23 -> units/d/a -> air_factory.png (200x63, 5 frames of 40x63)
    "airFactory": {
        "image": f"{DRAWABLE_DIR}/air_factory.png",
        "frames": 5,
        "scale": 1.0,
        "kind": "building",
        "evidence": "ar$23 -> units/d/a",
    },
    # ar$34 -> units/d/t -> sea_factory.png (90x99)
    "seaFactory": {
        "image": f"{DRAWABLE_DIR}/sea_factory.png",
        "frames": 1,
        "scale": 1.0,
        "kind": "building",
        "evidence": "ar$34 -> units/d/t",
    },
    # ar$2 -> units/e/g -> hover_tank.png (17x24)
    "hoverTank": {
        "image": f"{DRAWABLE_DIR}/hover_tank.png",
        "frames": 1,
        "scale": 1.0,
        "kind": "mobile",
        "evidence": "ar$2 -> units/e/g",
    },
    # ar$11 -> units/e/i -> hovercraft.png (20x32)
    "hovercraft": {
        "image": f"{DRAWABLE_DIR}/hovercraft.png",
        "frames": 1,
        "scale": 1.0,
        "kind": "mobile",
        "evidence": "ar$11 -> units/e/i",
    },
    # ar$19 -> units/b/d -> dropship.png (45x47)
    "dropship": {
        "image": f"{DRAWABLE_DIR}/dropship.png",
        "frames": 1,
        "scale": 1.0,
        "kind": "mobile",
        "evidence": "ar$19 -> units/b/d",
    },
    # ar$8 -> units/h/c -> gun_boat.png (15x27)
    "gunBoat": {
        "image": f"{DRAWABLE_DIR}/gun_boat.png",
        "frames": 1,
        "scale": 1.0,
        "kind": "mobile",
        "evidence": "ar$8 -> units/h/c",
    },
    # ar$6 -> units/b/e -> gunship.png (25x35)
    "gunShip": {
        "image": f"{DRAWABLE_DIR}/gunship.png",
        "frames": 1,
        "scale": 1.0,
        "kind": "mobile",
        "evidence": "ar$6 -> units/b/e",
    },
    # ar$14 -> units/h/a -> battle_ship_t2.png (17x68) + battle_ship_t2_turret.png
    "battleShip": {
        "image": f"{DRAWABLE_DIR}/battle_ship_t2.png",
        "frames": 1,
        "scale": 1.0,
        "kind": "mobile",
        "turretImage": f"{DRAWABLE_DIR}/battle_ship_t2_turret.png",
        "turretFrames": 1,
        "turretScale": 1.0,
        "evidence": "ar$14 -> units/h/a",
    },
    # ar$7 -> units/h/d -> scout_ship.png (17x31)  (the classic missile boat art)
    "missileShip": {
        "image": f"{DRAWABLE_DIR}/scout_ship.png",
        "frames": 1,
        "scale": 1.0,
        "kind": "mobile",
        "evidence": "ar$7 -> units/h/d",
    },
}

#: ``unit_type`` values that the game/replay layer emits but which name a unit
#: differently in the ini tree. Two groups:
#:
#: * **legacy pseudo-types** written by older recorders (``scripts/play_rwpp_policy.py``
#:   and ``scripts/record_rwpp_replay.py`` fall back to ``tank``/``factory``/
#:   ``command_center``/``building`` when the bridge gives no ``unit_type``);
#: * **snake_case spellings** that the Java bridge normalises via
#:   ``resolveUnitTypeName()`` (``land_factory`` -> ``landFactory``), but which
#:   can still appear in old artefacts.
ALIASES: dict[str, str] = {
    # real-game short names -> ini names
    "amphibiousJet": "c_amphibiousJet",
    "tank": "c_tank",
    "interceptor": "c_interceptor",
    "helicopter": "c_helicopter",
    "laserTank": "c_laserTank",
    "mammothTank": "c_mammothTank",
    "experimentalTank": "c_experimentalTank",
    # legacy recorder pseudo-types
    "command_center": "commandCenter",
    "factory": "landFactory",
    # snake_case spellings of the camelCase classics
    "land_factory": "landFactory",
    "air_factory": "airFactory",
    "sea_factory": "seaFactory",
    "command_center_snake": "commandCenter",
    "hover_tank": "hoverTank",
    "gun_boat": "gunBoat",
    "gun_ship": "gunShip",
    "battle_ship": "battleShip",
    "missile_ship": "missileShip",
    "scout_ship": "missileShip",
    "light_gunship": "lightGunship",
    "mech_factory": "mechFactory",
    "combat_engineer": "combatEngineer",
}


def _fallback_entries(root: Path) -> dict[str, dict[str, Any]]:
    """The three always-present last-resort sprites (contract §1 ``fallback``).

    Deliberately chosen so that a fallback still looks plausible: a tank for
    unknown mobile units, the builder for unknown builders, and a turret base
    for unknown buildings. Sizes match the contract's example block.
    """

    mobile = _sprite_entry(
        root, f"{GAME_UNITS_DIR}/tanks/tank.png", 3, 0.8, "mobile", "fallback mobile"
    )
    builder = _sprite_entry(
        root, f"{DRAWABLE_DIR}/builder.png", 1, 1.0, "mobile", "fallback builder"
    )
    building = _sprite_entry(
        root, f"{DRAWABLE_DIR}/turret_base.png", 1, 1.0, "building", "fallback building"
    )
    return {"mobile": mobile, "builder": builder, "building": building}


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def build_manifest(assets_root: Path | str | None = None) -> dict[str, Any]:
    """Produce the ``unitSprites`` manifest described by the contract.

    Raises :class:`UnitSpriteError` if any referenced image is missing or if a
    sheet cannot be cut into ``total_frames`` frames — never returns a manifest
    that would render as garbage.
    """

    root = Path(assets_root) if assets_root is not None else _assets_root()
    game_root = root / GAME_UNITS_DIR
    if not game_root.is_dir():
        raise UnitSpriteError(
            f"unit asset cache missing: {game_root} "
            "(run web/scripts/prepare_assets.ps1)"
        )

    by_type: dict[str, dict[str, Any]] = {}

    # 1. ini units first: their names are authoritative (c_tank, c_turret_t1 …).
    for unit in load_ini_units(game_root):
        if not unit.name or unit.name.upper() == "NULL":
            continue  # abstract common/base ini, not a spawnable unit
        entry = _entry_from_ini(unit, root, game_root)
        if entry is None:
            continue
        # An ini name later in the walk must not silently shadow an earlier one.
        by_type.setdefault(unit.name, entry)

    # 2. legacy builtin types, only where no ini claimed the name.
    for name, spec in CLASSIC_SPRITES.items():
        if name in by_type:
            continue
        entry = _sprite_entry(
            root,
            spec["image"],
            spec["frames"],
            spec["scale"],
            spec["kind"],
            f"classic {name} ({spec['evidence']})",
        )
        turret_image = spec.get("turretImage")
        if turret_image:
            turret = _sprite_entry(
                root,
                turret_image,
                spec.get("turretFrames", 1),
                spec.get("turretScale", spec["scale"]),
                spec["kind"],
                f"classic {name} turret ({spec['evidence']})",
            )
            entry["turretImage"] = turret["image"]
            entry["turretFrames"] = turret["frames"]
            entry["turretFrameW"] = turret["frameW"]
            entry["turretFrameH"] = turret["frameH"]
            entry["turretScale"] = turret["scale"]
        by_type[name] = entry

    # 3. Aliases must point at something real; a dangling alias is a bug.
    for alias, target in ALIASES.items():
        if target not in by_type:
            raise UnitSpriteError(
                f"alias {alias!r} -> {target!r} does not resolve to a sprite entry"
            )
        if alias in by_type:
            raise UnitSpriteError(f"alias {alias!r} collides with a real unit type")

    sorted_by_type = {key: by_type[key] for key in sorted(by_type)}
    return {
        "version": MANIFEST_VERSION,
        "byType": sorted_by_type,
        "aliases": dict(sorted(ALIASES.items())),
        "fallback": _fallback_entries(root),
    }


def unit_sprites_manifest(assets_root: Path | str | None = None) -> dict[str, Any]:
    """Small wrapper used by the ``/api/maps`` route.

    Returns an empty-but-valid manifest when the cache is absent so the endpoint
    keeps working on machines that never ran ``prepare_assets.ps1``.
    """

    try:
        return build_manifest(assets_root)
    except UnitSpriteError:
        return {
            "version": MANIFEST_VERSION,
            "byType": {},
            "aliases": {},
            "fallback": {},
        }


def resolve_type(manifest: Mapping[str, Any], unit_type: str | None) -> dict[str, Any] | None:
    """Reference implementation of contract §2.2, used by tests and probes."""

    if not unit_type:
        return None
    by_type = manifest.get("byType") or {}
    hit = by_type.get(unit_type)
    if hit:
        return hit
    aliases = manifest.get("aliases") or {}
    target = aliases.get(unit_type)
    if target:
        hit = by_type.get(target)
        if hit:
            return hit
    return None


def iter_manifest_entries(manifest: Mapping[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    """Yield ``(label, entry)`` for the body *and* turret of every entry."""

    for name, entry in (manifest.get("byType") or {}).items():
        yield name, entry
        if entry.get("turretImage"):
            yield f"{name}.turret", entry
    for name, entry in (manifest.get("fallback") or {}).items():
        yield f"fallback.{name}", entry
