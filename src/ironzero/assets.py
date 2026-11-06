# SPDX-License-Identifier: Apache-2.0
"""Read real Rusted Warfare 1.15 game data (units, maps, version) offline.

The micro-duel reference environment is calibrated from the *actual* game
assets whenever they are available, so the training dynamics match the real
client instead of invented numbers. Parsing lives here so the same code can
feed the asset manifest, the calibrated config and future scenario builders.

The parser only ever reads text files (``*.ini`` unit definitions, ``*.tmx``
map headers and ``lastrun.log``). No game binary is copied or redistributed.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .micro_config import MicroDuelConfig

# Rusted Warfare runs its fixed simulation loop at 60 ticks per second.
GAME_TICK_RATE = 60


class AssetError(ValueError):
    """Raised when the game asset layout does not look like Rusted Warfare."""


@dataclass(frozen=True, slots=True)
class UnitStats:
    key: str
    display_name: str
    price: float
    max_hp: float
    radius: float
    mass: float
    attack_range: float
    shoot_delay_ticks: int
    direct_damage: float
    projectile_speed: float
    move_speed_per_tick: float
    source: str

    @property
    def move_speed_per_second(self) -> float:
        return self.move_speed_per_tick * GAME_TICK_RATE

    @property
    def dps(self) -> float:
        if self.shoot_delay_ticks <= 0:
            return 0.0
        return self.direct_damage * GAME_TICK_RATE / self.shoot_delay_ticks

    def to_dict(self) -> dict[str, float | str]:
        return {
            "key": self.key,
            "display_name": self.display_name,
            "price": self.price,
            "max_hp": self.max_hp,
            "radius": self.radius,
            "mass": self.mass,
            "attack_range": self.attack_range,
            "shoot_delay_ticks": self.shoot_delay_ticks,
            "direct_damage": self.direct_damage,
            "projectile_speed": self.projectile_speed,
            "move_speed_per_tick": self.move_speed_per_tick,
            "move_speed_per_second": round(self.move_speed_per_second, 4),
            "dps": round(self.dps, 4),
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class MapInfo:
    name: str
    width_tiles: int
    height_tiles: int
    tile_width: int
    tile_height: int
    path: str
    sha256: str

    @property
    def width_px(self) -> int:
        return self.width_tiles * self.tile_width

    @property
    def height_px(self) -> int:
        return self.height_tiles * self.tile_height

    def to_dict(self) -> dict[str, int | str]:
        return {
            "name": self.name,
            "width_tiles": self.width_tiles,
            "height_tiles": self.height_tiles,
            "tile_width": self.tile_width,
            "tile_height": self.tile_height,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "path": self.path,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class GameInfo:
    game_dir: str
    version: str | None
    game_code: int | None
    build_number: str | None
    assets_dir: str
    assets_size_bytes: int


class GameAssets:
    """Lazy reader over a Rusted Warfare installation directory."""

    def __init__(self, game_dir: str | Path) -> None:
        self.root = Path(game_dir)
        self.assets = self.root / "assets"
        if not self.assets.is_dir():
            raise AssetError(f"assets/ not found under {self.root}")
        self.info = _read_game_info(self.root, self.assets)

    # -- units ----------------------------------------------------------
    def unit_ini_path(self, key: str) -> Path:
        candidate = self.assets / "units" / key
        if candidate.is_dir():
            ini = candidate / f"{key}.ini"
            if ini.is_file():
                return ini
            matches = sorted(candidate.glob("*.ini"))
            if matches:
                return matches[0]
        direct = self.assets / "units" / f"{key}.ini"
        if direct.is_file():
            return direct
        matches = sorted(self.assets.glob(f"units/**/{key}.ini"))
        if matches:
            return matches[0]
        raise AssetError(f"unit definition not found: {key}")

    def unit(self, key: str = "tank") -> UnitStats:
        path = self.unit_ini_path(key)
        return parse_unit_ini(path, key=key)

    def list_units(self) -> list[str]:
        units_dir = self.assets / "units"
        if not units_dir.is_dir():
            return []
        names: set[str] = set()
        for ini in units_dir.rglob("*.ini"):
            names.add(ini.stem)
        return sorted(names)

    # -- maps -----------------------------------------------------------
    def map_path(self, name: str = "[p2]Lake (2p).tmx") -> Path:
        candidate = self.assets / "maps" / "skirmish" / name
        if candidate.is_file():
            return candidate
        matches = sorted(self.assets.glob(f"maps/**/{name}"))
        if matches:
            return matches[0]
        for tmx in self.assets.glob("maps/**/*.tmx"):
            if name.lower() in tmx.name.lower():
                return tmx
        raise AssetError(f"map not found: {name}")

    def map_info(self, name: str = "[p2]Lake (2p).tmx") -> MapInfo:
        path = self.map_path(name)
        return parse_tmx_header(path)

    # -- calibration ----------------------------------------------------
    def calibrated_micro_config(
        self,
        *,
        unit_key: str = "tank",
        arena_size: float = 1000.0,
        units_per_team: int = 8,
        groups: int = 4,
        episode_seconds: float = 60.0,
        decisions_per_second: float = 2.0,
        spawn_margin: float = 220.0,
    ) -> MicroDuelConfig:
        unit = self.unit(unit_key)
        decision_interval = max(1, int(round(GAME_TICK_RATE / decisions_per_second)))
        episode_ticks = int(round(episode_seconds * GAME_TICK_RATE))
        return MicroDuelConfig(
            tick_rate=GAME_TICK_RATE,
            decision_interval=decision_interval,
            episode_ticks=episode_ticks,
            map_width=arena_size,
            map_height=arena_size,
            spawn_margin=spawn_margin,
            units_per_team=units_per_team,
            groups=groups,
            unit_hp=unit.max_hp,
            unit_damage=unit.direct_damage,
            attack_range=unit.attack_range,
            attack_cooldown_ticks=max(1, unit.shoot_delay_ticks),
            unit_speed=unit.move_speed_per_second,
            unit_radius=unit.radius,
            unit_price=unit.price,
        )

    def manifest(self, *, unit_key: str = "tank", map_name: str = "[p2]Lake (2p).tmx") -> dict:
        unit = self.unit(unit_key)
        map_info = self.map_info(map_name)
        config = self.calibrated_micro_config(unit_key=unit_key)
        unit_payload = unit.to_dict()
        unit_payload["source"] = _relative_source(unit.source, self.root)
        map_payload = map_info.to_dict()
        map_payload["path"] = _relative_source(map_info.path, self.root)
        return {
            "schema": "ironzero.game_assets/1",
            "game": {
                "version": self.info.version,
                "game_code": self.info.game_code,
                "build_number": self.info.build_number,
                "assets_size_bytes": self.info.assets_size_bytes,
                "asset_file_count": _count_files(self.assets),
            },
            "tick_rate": GAME_TICK_RATE,
            "unit": unit_payload,
            "map": map_payload,
            "micro_config": {
                "tick_rate": config.tick_rate,
                "decision_interval": config.decision_interval,
                "episode_ticks": config.episode_ticks,
                "map_width": config.map_width,
                "map_height": config.map_height,
                "spawn_margin": config.spawn_margin,
                "units_per_team": config.units_per_team,
                "groups": config.groups,
                "unit_hp": config.unit_hp,
                "unit_damage": config.unit_damage,
                "attack_range": config.attack_range,
                "attack_cooldown_ticks": config.attack_cooldown_ticks,
                "unit_speed": config.unit_speed,
                "unit_radius": config.unit_radius,
                "unit_price": config.unit_price,
            },
        }


# ----------------------------------------------------------------------
# Parsing helpers
# ----------------------------------------------------------------------
def parse_unit_ini(path: str | Path, *, key: str | None = None) -> UnitStats:
    path = Path(path)
    sections = _parse_ini(path.read_text(encoding="utf-8", errors="replace"))
    core = sections.get("core", {})
    attack = sections.get("attack", {})
    movement = sections.get("movement", {})
    projectile = _first_section(sections, "projectile")
    turret = _first_section(sections, "turret")
    if not core:
        raise AssetError(f"{path} has no [core] section")
    display_name = core.get("displayLocaleKey") or core.get("name") or path.stem
    shoot_delay = _as_int(turret.get("shootDelay") or attack.get("shootDelay"), default=60)
    damage = _as_float(projectile.get("directDamage"), default=0.0)
    return UnitStats(
        key=key or path.stem,
        display_name=display_name,
        price=_as_float(core.get("price"), default=0.0),
        max_hp=_as_float(core.get("maxHp"), default=100.0),
        radius=_as_float(core.get("radius"), default=8.0),
        mass=_as_float(core.get("mass"), default=0.0),
        attack_range=_as_float(attack.get("maxAttackRange") or turret.get("maxAttackRange"), default=100.0),
        shoot_delay_ticks=shoot_delay,
        direct_damage=damage,
        projectile_speed=_as_float(projectile.get("speed"), default=0.0),
        move_speed_per_tick=_as_float(movement.get("moveSpeed"), default=1.0),
        source=str(path),
    )


def parse_tmx_header(path: str | Path) -> MapInfo:
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"<map\b[^>]*>", text)
    if not match:
        raise AssetError(f"{path} is not a Tiled map")
    header = match.group(0)
    attrs = dict(re.findall(r'(\w+)="([^"]*)"', header))
    return MapInfo(
        name=path.name,
        width_tiles=int(attrs.get("width", 0)),
        height_tiles=int(attrs.get("height", 0)),
        tile_width=int(attrs.get("tilewidth", 0)),
        tile_height=int(attrs.get("tileheight", 0)),
        path=str(path),
        sha256=_sha256(path),
    )


def _read_game_info(root: Path, assets: Path) -> GameInfo:
    version = game_code = build_number = None
    log = root / "lastrun.log"
    if log.is_file():
        text = log.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"Game Version:\s*([0-9.]+)", text)
        version = match.group(1) if match else None
        match = re.search(r"Game Code:\s*(\d+)", text)
        game_code = int(match.group(1)) if match else None
        match = re.search(r"Build Number:\s*#?(\d+)", text)
        build_number = match.group(1) if match else None
    size = sum(item.stat().st_size for item in assets.rglob("*") if item.is_file())
    return GameInfo(
        game_dir=str(root),
        version=version,
        game_code=game_code,
        build_number=build_number,
        assets_dir=str(assets),
        assets_size_bytes=size,
    )


def _parse_ini(text: str) -> dict[str, dict[str, str]]:
    sections: dict[str, dict[str, str]] = {}
    current: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = sections.setdefault(line[1:-1].strip().lower(), {})
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        current[key.strip()] = value.strip()
    return sections


def _first_section(sections: dict[str, dict[str, str]], prefix: str) -> dict[str, str]:
    for name, values in sections.items():
        if name == prefix or name.startswith(prefix + "_"):
            return values
    return {}


def _as_float(value: str | None, *, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _as_int(value: str | None, *, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _relative_source(path: str | Path, root: Path) -> str:
    """Report asset provenance relative to the game directory (no local paths)."""

    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return Path(path).name


def _count_files(root: Path) -> int:
    return sum(1 for item in root.rglob("*") if item.is_file())


def load_game_assets(game_dir: str | Path) -> GameAssets:
    return GameAssets(game_dir)


def find_game_dir(candidates: Iterable[str | Path]) -> str:
    for candidate in candidates:
        if candidate and Path(candidate).is_dir() and (Path(candidate) / "assets").is_dir():
            return str(candidate)
    raise AssetError("no Rusted Warfare installation found in the candidate paths")


__all__ = [
    "AssetError",
    "GAME_TICK_RATE",
    "GameAssets",
    "GameInfo",
    "MapInfo",
    "UnitStats",
    "find_game_dir",
    "load_game_assets",
    "parse_tmx_header",
    "parse_unit_ini",
]
