# SPDX-License-Identifier: Apache-2.0
"""Action smoke test for the real RWPP bridge (stage 2).

Starts the real 1.15 headless game with the IronZero bridge, sends move /
attack-move / attack-target orders and verifies:

* receipts come back with ``accepted`` (actions reached the game command layer),
* own units actually move (positions change between observations).
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
    parser = argparse.ArgumentParser(description="RWPP action bridge smoke test")
    parser.add_argument("--port", type=int, default=5131)
    parser.add_argument("--ticks", type=int, default=1200)
    parser.add_argument("--timeout", type=float, default=150.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    env = dict(os.environ)
    env["JAVA_HOME"] = str(JAVA_HOME)
    env["PATH"] = str(JAVA_HOME / "bin") + os.pathsep + env.get("PATH", "")
    log_path = Path("runs/_rwpp_action_server.log")
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
        sock.settimeout(20)
        buffer = b""
        first_positions: dict[int, tuple[float, float]] = {}
        last_positions: dict[int, tuple[float, float]] = {}
        receipts = 0
        accepted = 0
        observations = 0
        enemy_ids: list[int] = []
        steps_sent = 0
        while time.time() < deadline:
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
                message = json.loads(line)
                kind = message.get("type")
                if kind == "obs":
                    observations += 1
                    units = message.get("own_entities", []) + message.get(
                        "visible_enemy_entities", []
                    )
                    for unit in message.get("own_entities", []):
                        last_positions[unit["entity_id"]] = (unit["x"], unit["y"])
                        first_positions.setdefault(unit["entity_id"], (unit["x"], unit["y"]))
                    for unit in message.get("visible_enemy_entities", []):
                        if unit["team_id"] == 1 and unit["entity_id"] not in enemy_ids:
                            enemy_ids.append(unit["entity_id"])
                    if observations == 1:
                        print("world:", len(message.get("own_entities", [])), "own,",
                              len(message.get("visible_enemy_entities", [])), "others",
                              "tick", message["tick"])
                    # send an order every few observations
                    if observations in (2, 6, 10) and steps_sent < 3:
                        steps_sent += 1
                        if steps_sent == 1:
                            actions = [{"kind": "move_group", "unit_group": 0, "x": 1500.0, "y": 1200.0}]
                        elif steps_sent == 2:
                            actions = [{"kind": "attack_move", "unit_group": 0, "x": 1900.0, "y": 900.0}]
                        else:
                            target = enemy_ids[0] if enemy_ids else None
                            if target is None:
                                continue
                            actions = [{"kind": "attack_target", "unit_group": 0, "target_entity_id": target}]
                        step = {
                            "protocol": "ironzero/1",
                            "type": "step",
                            "message_id": f"m-{steps_sent}",
                            "episode_id": "rwpp-bridge",
                            "tick": message["tick"],
                            "decision_id": f"d-{steps_sent}",
                            "actions": actions,
                        }
                        sock.sendall((json.dumps(step) + "\n").encode("utf-8"))
                elif kind == "receipt":
                    receipts += 1
                    if message.get("accepted"):
                        accepted += 1
                    print(
                        "receipt accepted=", message.get("accepted"),
                        "executed_at_tick=", message.get("executed_at_tick"),
                        "decision=", message.get("decision_id"),
                    )
                elif kind == "done":
                    print("done tick", message.get("tick"))
                    break
        moved = 0
        max_move = 0.0
        for entity_id, start in first_positions.items():
            if entity_id not in last_positions:
                continue
            end = last_positions[entity_id]
            distance = ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5
            if distance > 5.0:
                moved += 1
            max_move = max(max_move, distance)
        print(
            f"observations={observations} receipts={receipts} accepted={accepted} "
            f"units_moved={moved}/{len(first_positions)} max_move={max_move:.1f}px"
        )
        ok = receipts > 0 and accepted > 0 and moved > 0
        print("ACTION BRIDGE", "PASS" if ok else "FAIL")
        try:
            sock.sendall(b'{"type":"shutdown"}\n')
        except OSError:
            pass
        return 0 if ok else 1
    finally:
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
        log.close()


if __name__ == "__main__":
    sys.exit(main())
