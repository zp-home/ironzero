# SPDX-License-Identifier: Apache-2.0
"""Decode a TMX map's tile layers and report which tiles are placed.

Used to verify (rather than guess) whether a stock skirmish map actually places
resource pools: ``res_pool`` lives in *tileset* properties, so the only way to
know if a map has ore is to decode the Items layer and resolve every gid
against its tileset.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def decode_layer(layer: ET.Element, width: int, height: int) -> list[int]:
    data = layer.find("data")
    if data is None:
        return [0] * (width * height)
    encoding = data.get("encoding", "")
    compression = data.get("compression", "")
    raw = (data.text or "").strip()
    if encoding != "base64":
        raise SystemExit(f"unsupported encoding: {encoding!r}")
    payload = base64.b64decode(raw)
    if compression == "gzip":
        payload = gzip.decompress(payload)
    elif compression == "zlib":
        import zlib

        payload = zlib.decompress(payload)
    if len(payload) % 4:
        raise SystemExit(f"bad payload length {len(payload)}")
    values = [
        int.from_bytes(payload[i : i + 4], "little", signed=False)
        for i in range(0, len(payload), 4)
    ]
    if len(values) != width * height:
        raise SystemExit(f"layer size mismatch {len(values)} != {width * height}")
    return values


def load_tilesets(root: ET.Element, assets_root: Path) -> list[dict]:
    tilesets: list[dict] = []
    for element in root.findall("tileset"):
        first_gid = int(element.get("firstgid"))
        source = element.get("source")
        props: dict[int, dict[str, str]] = {}
        name = element.get("name") or ""
        if source:
            tsx = assets_root / "tilesets" / source
            if tsx.is_file():
                sub = ET.parse(tsx).getroot()
                name = sub.get("name") or name
                for tile in sub.findall("tile"):
                    tile_id = int(tile.get("id"))
                    entry: dict[str, str] = {}
                    for prop in tile.findall("properties/property"):
                        entry[prop.get("name")] = prop.get("value", "")
                    if entry:
                        props[tile_id] = entry
        else:
            for tile in element.findall("tile"):
                tile_id = int(tile.get("id"))
                entry = {}
                for prop in tile.findall("properties/property"):
                    entry[prop.get("name")] = prop.get("value", "")
                if entry:
                    props[tile_id] = entry
        tilesets.append({"first_gid": first_gid, "name": name, "props": props})
    tilesets.sort(key=lambda t: t["first_gid"])
    return tilesets


def resolve(gid: int, tilesets: list[dict]) -> tuple[str, int, dict[str, str]] | None:
    if gid == 0:
        return None
    flags = gid & 0xF0000000
    raw = gid & ~0xF0000000
    chosen = None
    for index, ts in enumerate(tilesets):
        next_first = tilesets[index + 1]["first_gid"] if index + 1 < len(tilesets) else None
        if raw >= ts["first_gid"] and (next_first is None or raw < next_first):
            chosen = ts
            break
    if chosen is None:
        return None
    local = raw - chosen["first_gid"]
    return chosen["name"], local, chosen["props"].get(local, {})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--map",
        default=r"E:\IronZeroBuild\rwpp-runtime-1.15\assets\maps\skirmish\[p2]Lake (2p).tmx",
    )
    parser.add_argument(
        "--assets",
        default=r"E:\IronZeroBuild\rwpp-runtime-1.15\assets",
    )
    args = parser.parse_args(argv)

    map_path = Path(args.map)
    assets_root = Path(args.assets)
    if not map_path.is_file():
        print(f"map not found: {map_path}", file=sys.stderr)
        return 2

    root = ET.parse(map_path).getroot()
    width = int(root.get("width"))
    height = int(root.get("height"))
    tilesets = load_tilesets(root, assets_root)
    print(f"map       = {map_path.name}")
    print(f"size      = {width} x {height}")
    print(f"tilesets  = " + ", ".join(f"{t['name']}@{t['first_gid']}" for t in tilesets))

    for layer in root.findall("layer"):
        name = layer.get("name")
        values = decode_layer(layer, width, height)
        non_zero = [v for v in values if v]
        print(f"\nlayer {name}: {len(non_zero)} non-zero tiles")
        histogram: dict[tuple[str, int, str], int] = {}
        for gid in non_zero:
            resolved = resolve(gid, tilesets)
            if resolved is None:
                histogram[("<unresolved>", gid, "")] = histogram.get(("<unresolved>", gid, ""), 0) + 1
                continue
            ts_name, local, props = resolved
            unit = props.get("unit", "")
            key = (ts_name, local, unit)
            histogram[key] = histogram.get(key, 0) + 1
            if "res_pool" in props:
                histogram[("<RES_POOL>", local, unit)] = histogram.get(("<RES_POOL>", local, unit), 0) + 1
        for (ts_name, local, unit), count in sorted(histogram.items(), key=lambda kv: -kv[1]):
            suffix = f" unit={unit}" if unit else ""
            print(f"  {ts_name} tile {local:>3} x{count}{suffix}")

    res_total = 0
    for layer in root.findall("layer"):
        values = decode_layer(layer, width, height)
        for gid in values:
            resolved = resolve(gid, tilesets)
            if resolved and "res_pool" in resolved[2]:
                res_total += 1
    print(f"\nres_pool tiles placed on this map: {res_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
