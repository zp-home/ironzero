# SPDX-License-Identifier: Apache-2.0
"""Raw-protocol probe for the IronZero 25-vs-25 tank arena.

Unlike ``baseline_rwpp.py`` this script deliberately bypasses the Python RL
environment: it speaks ``ironzero/1`` NDJSON to the bridge directly. That makes
it the lowest-level check that the fixed map, the annihilation win condition and
the AI unlock are all working, and it gives a scripted-policy win rate that the
trained PPO model must beat.

Example::

    $env:PYTHONPATH = "src"
    python scripts/probe_ironzero25.py --episodes 10 --strategy enemy_centroid
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME = Path(os.environ.get("IRONZERO_RWPP_RUNTIME", r"E:\IronZeroBuild\rwpp-runtime-1.15"))
DEFAULT_JAVA_HOME = Path(os.environ.get("IRONZERO_JAVA_HOME", r"E:\jdk25copy"))
DEFAULT_MAP = "maps/ironzero/IronZero25.tmx"
FULL_TEAM_HP = 25 * 210.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raw bridge probe for IronZero25")
    parser.add_argument("--runtime", default=str(DEFAULT_RUNTIME),
                        help="ASCII runtime path (Chinese paths break the RWML ASM agent)")
    parser.add_argument("--java-home", default=str(DEFAULT_JAVA_HOME))
    parser.add_argument("--map", default=DEFAULT_MAP)
    parser.add_argument("--map-width", type=float, default=2700.0)
    parser.add_argument("--map-height", type=float, default=2500.0)
    parser.add_argument("--ticks", type=int, default=5400)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--frame-ms", type=int, default=1)
    parser.add_argument("--port-base", type=int, default=5450)
    parser.add_argument("--strategy", default="enemy_centroid",
                        choices=["enemy_centroid", "nearest", "lowhp", "first"])
    parser.add_argument("--out", default="runs/probe_ironzero25")
    return parser.parse_args()


def start_game(runtime: Path, java_home: Path, port: int, ticks: int, map_relative: str, frame_ms: int):
    env = dict(os.environ)
    env["JAVA_HOME"] = str(java_home)
    args = [
        str(runtime / "bin" / "rwpp-server.bat"),
        "--no-console",
        "--",
        "--bridge-port",
        str(port),
        "--ticks",
        str(int(ticks)),
        "--map",
        map_relative,
        "--frame-ms",
        str(max(0, int(frame_ms))),
        "--win-mode",
        "annihilation",
        "--unlock-ai",
    ]
    return subprocess.Popen(
        args,
        cwd=str(runtime),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def connect(port: int, timeout: float = 60.0) -> socket.socket:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            sock = socket.create_connection(("127.0.0.1", port), timeout=2.0)
            sock.settimeout(180.0)
            return sock
        except OSError:
            time.sleep(0.2)
    raise RuntimeError(f"bridge port {port} never opened")


def centroid(units: list[dict[str, Any]]) -> tuple[float, float]:
    if not units:
        return (0.0, 0.0)
    return (
        sum(float(unit.get("x") or 0.0) for unit in units) / len(units),
        sum(float(unit.get("y") or 0.0) for unit in units) / len(units),
    )


def pick_action(obs: dict[str, Any], strategy: str) -> dict[str, Any]:
    own = obs.get("own_entities") or []
    enemies = obs.get("visible_enemy_entities") or []
    if not enemies:
        return {"kind": "attack_move", "unit_group": 4, "x": 1350.0, "y": 1250.0}
    own_cx, own_cy = centroid(own)
    if strategy == "nearest":
        target = min(enemies, key=lambda u: (float(u["x"]) - own_cx) ** 2 + (float(u["y"]) - own_cy) ** 2)
    elif strategy == "lowhp":
        target = min(
            enemies,
            key=lambda u: (float(u.get("hp") or 0.0), (float(u["x"]) - own_cx) ** 2 + (float(u["y"]) - own_cy) ** 2),
        )
    elif strategy == "enemy_centroid":
        ex, ey = centroid(enemies)
        target = min(enemies, key=lambda u: (float(u["x"]) - ex) ** 2 + (float(u["y"]) - ey) ** 2)
    else:
        target = enemies[0]
    return {"kind": "attack_target", "unit_group": 4, "target_entity_id": int(target["entity_id"])}


def run_episode(args: argparse.Namespace, episode: int) -> dict[str, Any]:
    port = int(args.port_base) + int(episode)
    process = start_game(
        Path(args.runtime),
        Path(args.java_home),
        port,
        args.ticks,
        args.map,
        args.frame_ms,
    )
    started = time.time()
    samples: list[dict[str, Any]] = []
    decisions = 0
    result: str | None = None
    bridge_result: str | None = None
    first_damage_tick: int | None = None
    sock: socket.socket | None = None
    try:
        sock = connect(port)
        reader = sock.makefile("r", encoding="utf-8")
        for line in reader:
            try:
                message = json.loads(line)
            except ValueError:
                continue
            kind = message.get("type")
            if kind == "done":
                bridge_result = str(message.get("result"))
                if result is None:
                    result = "timeout" if bridge_result == "timeout" else bridge_result
                break
            if kind != "obs":
                continue
            own = message.get("own_entities") or []
            enemies = message.get("visible_enemy_entities") or []
            own_hp = sum(float(unit.get("hp") or 0.0) for unit in own)
            enemy_hp = sum(float(unit.get("hp") or 0.0) for unit in enemies)
            if first_damage_tick is None and (own_hp < FULL_TEAM_HP * 0.995 or enemy_hp < FULL_TEAM_HP * 0.995):
                first_damage_tick = int(message.get("tick") or 0)
            samples.append(
                {
                    "tick": int(message.get("tick") or 0),
                    "own_alive": len(own),
                    "enemy_alive": len(enemies),
                    "own_hp_frac": round(own_hp / FULL_TEAM_HP, 4),
                    "enemy_hp_frac": round(enemy_hp / FULL_TEAM_HP, 4),
                }
            )
            payload = {
                "protocol": "ironzero/1",
                "type": "step",
                "message_id": f"probe-{episode}-{decisions}",
                "episode_id": f"probe-{episode}",
                "tick": int(message.get("tick") or 0),
                "decision_id": f"d-{episode}-{decisions}",
                "actions": [pick_action(message, args.strategy)],
            }
            try:
                sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))
            except OSError:
                break
            decisions += 1
            if message.get("terminated") and message.get("result") in {"victory", "defeat"}:
                result = str(message.get("result"))
                bridge_result = result
                break
    finally:
        if sock is not None:
            try:
                sock.sendall(b'{"type":"shutdown"}\n')
                sock.close()
            except OSError:
                pass
        try:
            process.wait(timeout=15)
        except Exception:  # noqa: BLE001
            process.kill()
    last = samples[-1] if samples else {}
    return {
        "episode": episode,
        "result": result,
        "bridge_result": bridge_result,
        "decisions": decisions,
        "tick": last.get("tick"),
        "own_alive": last.get("own_alive"),
        "enemy_alive": last.get("enemy_alive"),
        "own_hp_frac": last.get("own_hp_frac"),
        "enemy_hp_frac": last.get("enemy_hp_frac"),
        "first_damage_tick": first_damage_tick,
        "seconds": round(time.time() - started, 1),
        "samples": samples[:: max(1, len(samples) // 8)] if samples else [],
    }


def main() -> int:
    args = parse_args()
    episodes = [run_episode(args, index) for index in range(int(args.episodes))]
    wins = sum(1 for episode in episodes if episode["result"] == "victory")
    losses = sum(1 for episode in episodes if episode["result"] == "defeat")
    timeouts = sum(1 for episode in episodes if episode["result"] == "timeout")
    summary = {
        "strategy": args.strategy,
        "episodes": len(episodes),
        "wins": wins,
        "losses": losses,
        "timeouts": timeouts,
        "win_rate": wins / max(1, len(episodes)),
        "mean_own_hp_frac": _mean(episode.get("own_hp_frac") for episode in episodes),
        "mean_enemy_hp_frac": _mean(episode.get("enemy_hp_frac") for episode in episodes),
        "mean_tick": _mean(episode.get("tick") for episode in episodes),
        "mean_seconds": _mean(episode.get("seconds") for episode in episodes),
        "runtime": args.runtime,
        "map": args.map,
        "ticks": args.ticks,
        "frame_ms": args.frame_ms,
    }
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "probe_report.json").write_text(
        json.dumps({"summary": summary, "episodes": episodes}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _mean(values) -> float:
    items = [float(value) for value in values if value is not None]
    return round(statistics.fmean(items), 4) if items else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
