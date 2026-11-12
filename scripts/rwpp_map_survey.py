# SPDX-License-Identifier: Apache-2.0
"""Survey which official maps are economically playable.

Answers: "can we use the stock maps to train the economy actions, or do we need
a custom map?"

For each candidate map the bridge is started once and the very first
observation's ``info.builders[].can_build`` is inspected. That list is produced
by the game engine itself (``OrderableUnit.canQueueAction``), so it reflects
real unit data instead of a guess from .tmx text.

Usage:
    $env:PYTHONPATH = "src"
    python scripts/rwpp_map_survey.py --limit 12
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

RUNTIME = Path(r"E:\IronZeroBuild\rwpp-runtime-1.15")
JAVA_HOME = Path(r"E:\jdk25copy")
ASSETS_MAPS = RUNTIME / "assets" / "maps"

# Folders that hold real playable skirmish/mission maps.
CANDIDATE_PREFIXES = ("skirmish", "challenge", "testing", "normal")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Survey maps for economy playability")
    p.add_argument("--port", type=int, default=5141)
    p.add_argument("--timeout", type=float, default=90.0)
    p.add_argument("--limit", type=int, default=0, help="0 = all candidates")
    p.add_argument("--need", default="extractor", help="unit type the builder must be able to make")
    return p


def list_maps(limit: int) -> list[str]:
    maps: list[str] = []
    for prefix in CANDIDATE_PREFIXES:
        folder = ASSETS_MAPS / prefix
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.tmx")):
            rel = path.relative_to(RUNTIME / "assets").as_posix()
            maps.append(rel)
    return maps[:limit] if limit else maps


def probe(map_rel: str, port: int, timeout: float) -> dict:
    env = dict(os.environ)
    env["JAVA_HOME"] = str(JAVA_HOME)
    env["PATH"] = str(JAVA_HOME / "bin") + os.pathsep + env.get("PATH", "")
    log_path = Path("runs/_map_survey.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(
            [str(RUNTIME / "bin" / "rwpp-server.bat"), "--no-console", "--",
             "--bridge-port", str(port), "--ticks", "240"], 
            cwd=str(RUNTIME), env=env, stdout=log, stderr=subprocess.STDOUT,
        )
        sock = None
        try:
            deadline = time.time() + timeout
            while time.time() < deadline:
                try:
                    sock = socket.create_connection(("127.0.0.1", port), timeout=2)
                    break
                except OSError:
                    time.sleep(0.4)
            if sock is None:
                return {"map": map_rel, "error": "bridge never opened"}
            sock.settimeout(20)
            buf = b""
            while time.time() < deadline:
                try:
                    chunk = sock.recv(1 << 16)
                except socket.timeout:
                    break
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line.strip():
                        continue
                    msg = json.loads(line)
                    if msg.get("type") == "obs":
                        info = msg.get("info") or {}
                        return {
                            "map": map_rel,
                            "builders": info.get("builders") or [],
                            "own": len(msg.get("own_entities") or []),
                            "units_alive": info.get("units_alive"),
                        }
            return {"map": map_rel, "error": "no obs"}
        finally:
            if sock is not None:
                try:
                    sock.sendall(b'{"type":"shutdown"}\n')
                except OSError:
                    pass
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                process.kill()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    maps = list_maps(args.limit)
    print(f"surveying {len(maps)} candidate maps for buildable '{args.need}'\n")
    playable: list[str] = []
    for index, rel in enumerate(maps, 1):
        result = probe(rel, args.port, args.timeout)
        if "error" in result:
            print(f"[{index:2d}/{len(maps)}] {rel}  -> {result['error']}")
            continue
        builders = result.get("builders") or []
        can = sorted({t for b in builders for t in (b.get("can_build") or [])})
        types = sorted({b.get("unit_type") for b in builders})
        hit = args.need in can
        if hit:
            playable.append(rel)
        mark = "YES" if hit else " no"
        print(f"[{index:2d}/{len(maps)}] {mark}  {rel}")
        print(f"        builders={types} can_build={can}")

    print()
    print(f"maps where a builder can make '{args.need}': {len(playable)}")
    for rel in playable:
        print("   ", rel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
