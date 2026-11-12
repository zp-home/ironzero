# SPDX-License-Identifier: Apache-2.0
"""Dump the first real-game observation so the entity stream can be audited.

Training metrics are aggregated and can hide the shape of the world (for example
"28 enemies" on a map that ships 2). This probe prints the raw entity lists, the
unit-type histogram and the ``global_state`` counters, then exits -- no actions
are sent, so it is safe to run next to a training job on another port.

Run::

    python scripts/rwpp_obs_probe.py --map "maps/skirmish/[p2]Lake (2p).tmx"

Exit codes: 0 = stream read, 2 = bridge never opened, 1 = protocol error.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ironzero.protocol import parse_message  # noqa: E402

RUNTIME = Path(os.environ.get("IRONZERO_RWPP_RUNTIME", r"E:\IronZeroBuild\rwpp-runtime-1.15"))
JAVA_HOME = Path(os.environ.get("IRONZERO_JAVA_HOME", r"E:\jdk25copy"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dump first RWPP observation")
    parser.add_argument("--port", type=int, default=5333)
    parser.add_argument("--ticks", type=int, default=900)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--observations", type=int, default=2, help="how many obs to read")
    parser.add_argument("--map", default="maps/skirmish/[p2]Lake (2p).tmx")
    parser.add_argument("--unlock-ai", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    env = dict(os.environ)
    env["JAVA_HOME"] = str(JAVA_HOME)
    env["PATH"] = str(JAVA_HOME / "bin") + os.pathsep + env.get("PATH", "")
    log_path = REPO / "runs" / "_rwpp_obs_probe.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(RUNTIME / "bin" / "rwpp-server.bat"), "--no-console", "--",
        "--bridge-port", str(args.port),
        "--ticks", str(args.ticks),
        "--map", args.map,
    ]
    if args.unlock_ai:
        command.append("--unlock-ai")
    log = log_path.open("w", encoding="utf-8", errors="replace")
    process = subprocess.Popen(command, cwd=str(RUNTIME), env=env,
                               stdout=log, stderr=subprocess.STDOUT)
    sock = None
    deadline = time.time() + args.timeout
    try:
        while time.time() < deadline:
            try:
                sock = socket.create_connection(("127.0.0.1", args.port), timeout=2)
                break
            except OSError:
                time.sleep(0.5)
        if sock is None:
            print("bridge port never opened; see", log_path, file=sys.stderr)
            return 2

        sock.settimeout(30)
        buffer = b""
        seen = 0
        errors: list[str] = []
        while time.time() < deadline and seen < args.observations:
            try:
                chunk = sock.recv(1 << 16)
            except socket.timeout:
                break
            if not chunk:
                break
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if not line.strip():
                    continue
                text = line.decode("utf-8", errors="replace")
                try:
                    parse_message(text)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{type(exc).__name__}: {str(exc)[:160]}")
                message = json.loads(text)
                if message.get("type") != "obs":
                    continue
                seen += 1
                own = message.get("own_entities", [])
                enemies = message.get("visible_enemy_entities", [])
                print(f"--- obs #{seen} tick={message.get('tick')} ---")
                print("global_state:", json.dumps(message.get("global_state"), ensure_ascii=False))
                print("info:", json.dumps(message.get("info"), ensure_ascii=False)[:400])
                print(f"own    n={len(own)}  types={dict(Counter(u.get('unit_type') for u in own))}")
                print(f"enemy  n={len(enemies)} types={dict(Counter(u.get('unit_type') for u in enemies))}")
                print("  enemy team_ids:", dict(Counter(u.get("team_id") for u in enemies)))
                print("  enemy is_building:", dict(Counter(bool(u.get("is_building")) for u in enemies)))
                print("  enemy sample:", json.dumps(enemies[:3], ensure_ascii=False))
                print("  own sample:", json.dumps(own[:3], ensure_ascii=False))
                print()
                if seen >= args.observations:
                    break

        if errors:
            print(f"protocol errors = {len(errors)}")
            for err in errors[:5]:
                print("   ", err)
            return 1
        return 0 if seen else 2
    finally:
        if sock is not None:
            try:
                sock.sendall(b'{"type":"shutdown"}\n')
            except OSError:
                pass
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
