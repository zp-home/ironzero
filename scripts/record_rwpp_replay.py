# SPDX-License-Identifier: Apache-2.0
"""Record a real Rusted Warfare 1.15 game (headless + RWPP bridge) as an IronZero replay.

This is the "real game" capture path:

    python scripts/record_rwpp_replay.py --ticks 1800

It starts the pinned RWPP headless runtime with the IronZero bridge extension,
reads ironzero/1 observations over loopback TCP, converts them into the
``ironzero.replay/1`` frame format and writes the replay under
``runs/platform/real-rwpp-*/replays/`` so the dashboard replay library lists it
(rendered on the real map by the sandbox).

The game runs one map with an AI opponent; actions are not injected yet.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

RUNTIME = Path(r"E:\IronZeroBuild\rwpp-runtime-1.15")
JAVA_HOME = Path(r"E:\jdk25copy")
MAP_RELATIVE = "maps/skirmish/[p2]Lake (2p).tmx"
MAP_NAME = "[p2]Lake (2p)"


def map_id_for(relative: str) -> str:
    return base64.urlsafe_b64encode(relative.encode("utf-8")).decode("ascii").rstrip("=")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record a real RWPP game as an IronZero replay")
    parser.add_argument("--port", type=int, default=5130)
    parser.add_argument("--ticks", type=int, default=1800)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--out-root", default="runs/platform")
    parser.add_argument("--map", default=MAP_RELATIVE)
    parser.add_argument("--map-name", default=MAP_NAME)
    parser.add_argument("--policy", choices=["none", "scripted"], default="scripted",
                        help="scripted = drive the player units with simple orders")
    parser.add_argument("--order-every", type=int, default=4, help="send an order every N observations")
    parser.add_argument("--stream", default="", help="optional JSONL path for live dashboard frames")
    parser.add_argument("--frame-ms", type=int, default=16,
                        help="wall-clock ms per simulated frame: 16 = real time, 0 = as fast as possible")
    parser.add_argument("--map-width", type=float, default=2600.0)
    parser.add_argument("--map-height", type=float, default=2600.0)
    return parser


def obs_to_frame(message: dict, config: dict) -> dict:
    units = []
    for entity in message.get("own_entities", []) + message.get("visible_enemy_entities", []):
        team = entity.get("team_id", -1)
        if team not in (0, 1):
            continue
        units.append(
            {
                "id": entity.get("entity_id"),
                "team": team,
                "group": 0,
                "x": entity.get("x", 0.0),
                "y": entity.get("y", 0.0),
                "hp": entity.get("hp", 0.0),
                "max_hp": entity.get("max_hp", 1.0),
                "order": "idle",
                "target": None,
            }
        )
    own_hp = sum(u["hp"] for u in units if u["team"] == 0)
    enemy_hp = sum(u["hp"] for u in units if u["team"] == 1)
    own_max = sum(u["max_hp"] for u in units if u["team"] == 0) or 1.0
    enemy_max = sum(u["max_hp"] for u in units if u["team"] == 1) or 1.0
    return {
        "episode": 0,
        "decision": message.get("tick", 0),
        "tick": message.get("tick", 0),
        "time": round(float(message.get("game_time", 0.0)), 3),
        "reward": 0.0,
        "units": units,
        "events": [],
        "stats": {
            "own_hp_frac": round(own_hp / own_max, 4),
            "enemy_hp_frac": round(enemy_hp / enemy_max, 4),
            "own_alive": sum(1 for u in units if u["team"] == 0),
            "enemy_alive": sum(1 for u in units if u["team"] == 1),
        },
        "result": None,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    env = dict(os.environ)
    env["JAVA_HOME"] = str(JAVA_HOME)
    env["PATH"] = str(JAVA_HOME / "bin") + os.pathsep + env.get("PATH", "")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    unique = uuid.uuid4().hex[:6]
    job_dir = Path(args.out_root) / f"real-rwpp-{stamp}-{unique}"
    replay_dir = job_dir / "replays"
    replay_dir.mkdir(parents=True, exist_ok=True)
    server_log = Path("runs") / f"_rwpp_record_{stamp}_{unique}.log"
    server_log.parent.mkdir(parents=True, exist_ok=True)
    log = server_log.open("w", encoding="utf-8", errors="replace")
    process = subprocess.Popen(
        [
            str(RUNTIME / "bin" / "rwpp-server.bat"),
            "--no-console",
            "--",
            "--bridge-port",
            str(args.port),
            "--ticks",
            str(args.ticks),
            "--map",
            args.map,
            "--frame-ms",
            str(args.frame_ms),
        ],
        cwd=str(RUNTIME),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    frames: list[dict] = []
    stream = None
    if args.stream:
        stream_path = Path(args.stream)
        stream_path.parent.mkdir(parents=True, exist_ok=True)
        stream = stream_path.open("w", encoding="utf-8")
        stream.write(json.dumps({"type": "start", "job": job_dir.name}) + "\n")
        stream.flush()
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
            print("bridge port never opened; see", server_log, file=sys.stderr)
            return 2
        sock.settimeout(30)
        buffer = b""
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
                    frame = obs_to_frame(message, {})
                    frames.append(frame)
                    if stream is not None:
                        stream.write(json.dumps({"type": "frame", **frame}, ensure_ascii=False) + "\n")
                        stream.flush()
                    if args.policy == "scripted" and len(frames) % max(1, args.order_every) == 0:
                        own = [u for u in message.get("own_entities", []) if u.get("team_id") == 0]
                        enemies = [u for u in message.get("visible_enemy_entities", [])
                                   if u.get("team_id") == 1]
                        actions = []
                        if own and enemies:
                            best = None
                            best_distance = float("inf")
                            for enemy in enemies:
                                for unit in own:
                                    distance = ((enemy["x"] - unit["x"]) ** 2
                                                + (enemy["y"] - unit["y"]) ** 2) ** 0.5
                                    if distance < best_distance:
                                        best_distance = distance
                                        best = enemy
                            actions = [{"kind": "attack_target", "unit_group": 0,
                                        "target_entity_id": best["entity_id"]}]
                        elif own:
                            actions = [{"kind": "attack_move", "unit_group": 0,
                                        "x": 1500.0, "y": 1500.0}]
                        if actions:
                            step = {
                                "protocol": "ironzero/1",
                                "type": "step",
                                "message_id": f"m-{len(frames)}",
                                "episode_id": "rwpp-bridge",
                                "tick": message.get("tick", 0),
                                "decision_id": f"d-{len(frames)}",
                                "actions": actions,
                            }
                            try:
                                sock.sendall((json.dumps(step) + "\n").encode("utf-8"))
                            except OSError:
                                pass
                elif kind == "done":
                    break
        print(f"captured {len(frames)} frames, server log {server_log}")
        if not frames:
            return 2
        meta = {
            "schema": "ironzero.replay/1",
            "checkpoint": "rwpp-live",
            "opponent": "ai",
            "seed": 0,
            "episodes": 1,
            "mode": "live",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sim_mode": "terrain",
            "map": {
                "id": map_id_for(args.map),
                "name": args.map_name,
                "width_px": args.map_width,
                "height_px": args.map_height,
                "tile_width": 20,
                "tile_height": 20,
            },
            "vision_radius": 260,
            "fog": False,
            "result": "timeout",
            "return": 0.0,
            "source": "rwpp-bridge",
        }
        replay_path = replay_dir / f"replay-rwpp-{stamp}-{unique}.json"
        replay_path.write_text(
            json.dumps(
                {"schema": "ironzero.replay/1", "meta": meta, "episodes": [{"episode": 0}], "frames": frames},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        (job_dir / "meta.json").write_text(
            json.dumps(
                {"id": job_dir.name, "name": "真机 RWPP 录制", "kind": "infer", "status": "succeeded",
                 "created_at": meta["created_at"], "config": {"sim_mode": "rwpp", "map": args.map_name}},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"replay written: {replay_path} ({len(frames)} frames)")
        if stream is not None:
            stream.write(
                json.dumps({"type": "done", "replay_path": str(replay_path), "frames": len(frames)}) + "\n"
            )
            stream.flush()
        try:
            sock.sendall(b'{"type":"shutdown"}\n')
        except OSError:
            pass
    finally:
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
        log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
