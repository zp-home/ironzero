# SPDX-License-Identifier: Apache-2.0
"""Smoke client for the IronZero RWPP bridge extension.

Starts the real 1.15 headless runtime with the bridge extension, connects to its
loopback NDJSON port and prints the first observations. This proves the real
game state path end-to-end (engine tick + units + teams) without the platform.
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RWPP bridge smoke client")
    parser.add_argument("--port", type=int, default=5130)
    parser.add_argument("--ticks", type=int, default=600)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--out", default="runs/_rwpp_bridge_smoke.jsonl")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    env = dict(os.environ)
    env["JAVA_HOME"] = str(JAVA_HOME)
    env["PATH"] = str(JAVA_HOME / "bin") + os.pathsep + env.get("PATH", "")
    log_path = Path("runs/_rwpp_bridge_server.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8", errors="replace")
    process = subprocess.Popen(
        [
            str(RUNTIME / "bin" / "rwpp-server.bat"),
            "--no-console",
            "--",
            "--bridge-port",
            str(args.port),
            "--ticks",
            str(args.ticks),
        ],
        cwd=str(RUNTIME),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + args.timeout
    sock = None
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
        sock.settimeout(10)
        buffer = b""
        observations = 0
        units = 0
        with out_path.open("w", encoding="utf-8") as out:
            while time.time() < deadline:
                try:
                    chunk = sock.recv(65536)
                except socket.timeout:
                    break
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if not line.strip():
                        continue
                    message = json.loads(line)
                    out.write(json.dumps(message, ensure_ascii=False) + "\n")
                    kind = message.get("type")
                    if kind == "hello":
                        print("hello:", message.get("adapter_version"), message.get("capabilities"))
                    elif kind == "obs":
                        observations += 1
                        units = len(message.get("own_entities", [])) + len(
                            message.get("visible_enemy_entities", [])
                        )
                        if observations in (1, 2, 5):
                            print(
                                f"obs tick={message['tick']} units={units} "
                                f"team0={len(message.get('own_entities', []))} "
                                f"others={len(message.get('visible_enemy_entities', []))}"
                            )
                    elif kind == "done":
                        print("done tick=", message.get("tick"))
                        break
                if observations >= 3 and time.time() > deadline - 1:
                    break
                if process.poll() is not None and not buffer:
                    break
        print(f"observations={observations} last_units={units} output={out_path}")
        try:
            sock.sendall(b'{"type":"shutdown"}\n')
        except OSError:
            pass
    finally:
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
        log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
