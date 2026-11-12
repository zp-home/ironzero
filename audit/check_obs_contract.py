# SPDX-License-Identifier: Apache-2.0
"""Diagnose the observation contract against a real economy map.

Answers, with numbers instead of guesses:

* how many own/enemy entities actually appear over an episode (does the
  ``max_entities`` slot budget truncate the battlefield?),
* which encoded features fall outside the ``[0, 1]`` range the network was
  trained to expect,
* how often the observation drops information the policy needs (buildings,
  builders) because slots are filled in raw ``Unit.unitList`` order.

Run against a live headless game:

    $env:PYTHONPATH = "src"
    python audit/check_obs_contract.py --ticks 4000
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ironzero.micro_config import DEFAULT_CONFIG  # noqa: E402
from ironzero.observations import encode_observation  # noqa: E402
from ironzero.rwpp.config import RwppGameConfig  # noqa: E402
from ironzero.rwpp.decoding import observation_to_state  # noqa: E402

DEFAULT_RUNTIME = Path(r"E:\IronZeroBuild\rwpp-runtime-1.15")
DEFAULT_JAVA_HOME = Path(r"E:\jdk25copy")
DEFAULT_MAP = "maps/skirmish/[p2]Lake (2p).tmx"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Observation contract diagnostic")
    parser.add_argument("--port", type=int, default=5139)
    parser.add_argument("--ticks", type=int, default=4000)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--map", default=DEFAULT_MAP)
    parser.add_argument("--runtime", default=str(DEFAULT_RUNTIME))
    parser.add_argument("--java-home", default=str(DEFAULT_JAVA_HOME))
    parser.add_argument("--frame-ms", type=int, default=1)
    parser.add_argument(
        "--units-per-team",
        type=int,
        default=2,
        help="the value the GUI currently pins for stock maps",
    )
    return parser


def start_game(args: argparse.Namespace) -> tuple[subprocess.Popen, socket.socket]:
    runtime = Path(args.runtime)
    java_home = Path(args.java_home)
    env = dict(os.environ)
    env["JAVA_HOME"] = str(java_home)
    env["PATH"] = str(java_home / "bin") + os.pathsep + env.get("PATH", "")
    log_path = Path("runs/_obs_contract.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8", errors="replace")
    process = subprocess.Popen(
        [
            str(runtime / "bin" / "rwpp-server.bat"),
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
        cwd=str(runtime),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    log.close()
    deadline = time.time() + args.timeout
    while time.time() < deadline:
        try:
            sock = socket.create_connection(("127.0.0.1", args.port), timeout=2.0)
            sock.settimeout(30.0)
            return process, sock
        except OSError:
            time.sleep(0.5)
    process.kill()
    raise SystemExit(f"bridge port {args.port} never opened; see {log_path}")


def read_obs(sock: socket.socket, buffer: bytes) -> tuple[dict | None, bytes]:
    while True:
        if b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            if not line.strip():
                continue
            try:
                message = json.loads(line)
            except ValueError:
                continue
            if message.get("type") in {"obs", "done"}:
                return message, buffer
            continue
        chunk = sock.recv(1 << 16)
        if not chunk:
            return None, buffer
        buffer += chunk


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Reproduce exactly what the GUI sends for a stock map.
    game = RwppGameConfig(
        map_relative=args.map,
        map_name="Lake",
        units_per_team=args.units_per_team,
        max_entities=DEFAULT_CONFIG.max_entities,
        target_slots=0,
        frame_ms=args.frame_ms,
        episode_ticks=args.ticks,
    )
    micro = game.micro_config()

    print("=== contract the GUI currently produces for this map ===")
    print(f"units_per_team = {micro.units_per_team}")
    print(f"max_entities   = {micro.max_entities}")
    print(f"target_slots   = {micro.target_slots}")
    print(f"flat_obs_dim   = {micro.flat_obs_dim}")
    print()

    process, sock = start_game(args)
    buffer = b""
    own_counts: list[int] = []
    enemy_counts: list[int] = []
    frames = 0
    worst: dict[str, float] = {}
    buildings_hidden = 0
    frames_with_buildings_outside = 0
    try:
        while frames < 200:
            obs, buffer = read_obs(sock, buffer)
            if obs is None:
                break
            if obs.get("type") == "done":
                break
            frames += 1
            state = observation_to_state(obs)
            own = state.get("own_entities", [])
            enemies = state.get("visible_enemy_entities", [])
            own_counts.append(len(own))
            enemy_counts.append(len(enemies))

            # Which own entities are buildings, and do they survive truncation?
            encoded_own = own[: micro.max_entities]
            dropped = [u for u in own[micro.max_entities :]]
            if dropped:
                frames_with_buildings_outside += 1
                buildings_hidden += sum(1 for u in dropped if u.get("is_building"))

            vector = encode_observation(state, micro)
            # Features 9/10 are signed direction deltas in [-1, 1]; every other
            # global feature is documented as normalised to [0, 1].
            for index in range(micro.global_dim):
                value = float(vector[index])
                if not np.isfinite(value):
                    continue
                low = worst.get(f"g{index}_min")
                if low is None or value < low:
                    worst[f"g{index}_min"] = value
                high = worst.get(f"g{index}")
                if high is None or value > high:
                    worst[f"g{index}"] = value

        print(f"frames sampled     = {frames}")
        print(f"own entities       = min {min(own_counts)} / max {max(own_counts)}"
              f"  (slot budget {micro.max_entities})")
        print(f"enemy entities     = min {min(enemy_counts)} / max {max(enemy_counts)}"
              f"  (slot budget {micro.max_entities})")
        print(f"frames where own entities were truncated = {frames_with_buildings_outside}")
        print(f"buildings dropped by truncation          = {buildings_hidden}")
        print()

        print("=== global-block features outside [0,1] ===")
        labels = {
            0: "tick/episode_ticks",
            1: "own_count/units_per_team",
            2: "enemy_count/units_per_team",
            3: "own_hp/total_hp",
            4: "enemy_hp/total_hp",
            5: "own_cx/map_width",
            6: "own_cy/map_height",
            7: "enemy_cx/map_width",
            8: "enemy_cy/map_height",
            9: "(enemy_cx-own_cx)/map_width",
            10: "(enemy_cy-own_cy)/map_height",
            11: "constant 1.0",
            12: "funds/funds_scale",
            13: "income/income_scale",
            14: "extractor_slots/count",
        }
        violations = 0
        # Features 9 and 10 are signed deltas, so their valid range is [-1, 1].
        signed = {9, 10}
        for index in range(micro.global_dim):
            low = worst.get(f"g{index}_min")
            high = worst.get(f"g{index}")
            if low is None or high is None:
                continue
            lower_bound = -1.0 if index in signed else 0.0
            if low < lower_bound - 1e-6 or high > 1.0 + 1e-6:
                violations += 1
                print(f"  [{index:2d}] {labels.get(index, '?'):<32} "
                      f"observed [{low:+.3f}, {high:+.3f}]  <-- OUT OF RANGE")
        if violations == 0:
            print("  (none)")
        print()
        print(f"global-block violations = {violations}")
        return 0 if violations == 0 else 1
    finally:
        try:
            sock.sendall(b'{"type":"shutdown"}\n')
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
