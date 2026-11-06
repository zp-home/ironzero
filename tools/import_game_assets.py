# SPDX-License-Identifier: Apache-2.0
"""Export a machine-readable manifest from a local Rusted Warfare installation.

Usage::

    python tools/import_game_assets.py --game-dir "E:/games/Rusted Warfare"

The manifest records only derived numbers (unit stats, map geometry, version),
never game binaries or asset files. It lets CI and teammates verify that the
environment was calibrated from the same 1.15 build.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ironzero.assets import AssetError, GameAssets  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import Rusted Warfare asset stats into a JSON manifest")
    parser.add_argument("--game-dir", required=True, help="directory containing assets/ and lastrun.log")
    parser.add_argument("--unit", default="tank", help="unit definition to import (default: tank)")
    parser.add_argument(
        "--map",
        default="[p2]Lake (2p).tmx",
        help="map file name under assets/maps (default: the M1 smoke map)",
    )
    parser.add_argument("--out", default=str(ROOT / "configs" / "game_assets.json"))
    parser.add_argument("--check", action="store_true", help="compare with the existing manifest and exit 1 on drift")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        assets = GameAssets(args.game_dir)
        manifest = assets.manifest(unit_key=args.unit, map_name=args.map)
    except AssetError as exc:
        print(f"asset import failed: {exc}", file=sys.stderr)
        return 2
    manifest["generated_by"] = "tools/import_game_assets.py"
    manifest["license_note"] = (
        "Derived numeric game data only; no Rusted Warfare binaries or assets are redistributed."
    )
    out = Path(args.out)
    payload = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    if args.check:
        if not out.exists():
            print(f"manifest missing: {out}", file=sys.stderr)
            return 1
        existing = json.loads(out.read_text(encoding="utf-8"))
        for key in ("game", "unit", "map", "micro_config"):
            if existing.get(key) != manifest.get(key):
                print(f"manifest drift in section: {key}", file=sys.stderr)
                return 1
        print(f"manifest up to date: {out}")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(payload, encoding="utf-8", newline="\n")
    unit = manifest["unit"]
    print(
        "imported {name}: hp={hp} dmg={dmg} range={range} cooldown={cd} ticks "
        "speed={speed} u/s (game {version} build #{build}, map {map} {w}x{h}px)".format(
            name=unit["key"],
            hp=unit["max_hp"],
            dmg=unit["direct_damage"],
            range=unit["attack_range"],
            cd=unit["shoot_delay_ticks"],
            speed=unit["move_speed_per_second"],
            version=manifest["game"]["version"],
            build=manifest["game"]["build_number"],
            map=manifest["map"]["name"],
            w=manifest["map"]["width_px"],
            h=manifest["map"]["height_px"],
        )
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
