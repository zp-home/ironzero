# SPDX-License-Identifier: Apache-2.0
"""Scripted-baseline evaluation on the real Rusted Warfare micro scenario.

This is the "is the scenario winnable at all?" probe used before spending GPU
time on PPO. It drives the same ``RwppDuelEnv`` contract as training, but the
policy is a tiny hand-written focus-fire controller:

* no visible enemy -> broadcast-attack-move towards the map centre;
* visible enemy   -> broadcast-attack-target the lowest-HP enemy.

If this baseline cannot win ~half of its games, the scenario or the weapon
numbers are wrong and training is pointless. It also doubles as a cheap
smoke-test that the whole bridge / observation / action path works for a
25-vs-25 tank battle.

Example::

    $env:PYTHONPATH = "src"
    python scripts/baseline_rwpp.py `
      --runtime ".runtime\\rwpp-runtime-1.15" `
      --map "maps/ironzero/IronZero25.tmx" --map-name "IronZero25" `
      --map-width 2700 --map-height 2500 `
      --units-per-team 25 --episodes 10 --ticks 3600 --frame-ms 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ironzero.micro_config import DEFAULT_CONFIG, MicroDuelConfig  # noqa: E402
from ironzero.rwpp_env import (  # noqa: E402
    RwppDuelEnv,
    RwppGameConfig,
    apply_vision,
)

BROADCAST = 4  # matches MicroDuelConfig.action_nvec[1] - 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scripted focus-fire baseline on real RWPP")
    parser.add_argument("--runtime", default=str(REPO_ROOT / ".runtime" / "rwpp-runtime-1.15"))
    parser.add_argument("--java-home", default=r"E:\jdk25copy")
    parser.add_argument("--map", default="maps/ironzero/IronZero25.tmx")
    parser.add_argument("--map-name", default="IronZero25")
    parser.add_argument("--map-width", type=float, default=2700.0)
    parser.add_argument("--map-height", type=float, default=2500.0)
    parser.add_argument("--units-per-team", type=int, default=25)
    parser.add_argument("--max-entities", type=int, default=32)
    parser.add_argument("--target-slots", type=int, default=26)
    parser.add_argument("--episode-ticks", type=int, default=3600)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--frame-ms", type=int, default=1)
    parser.add_argument("--vision-radius", type=float, default=0.0)
    parser.add_argument("--port-base", type=int, default=5430)
    parser.add_argument("--seed", type=int, default=31_000)
    parser.add_argument("--out", default="runs/baseline_rwpp")
    parser.add_argument("--terrain", default="configs/maps/IronZero25.terrain.json")
    parser.add_argument("--win-mode", default="annihilation", help="forwarded to bridge (empty = native)")
    parser.add_argument("--unlock-ai", action=argparse.BooleanOptionalAction, default=True,
                        help="let the built-in AI play (forwarded to bridge)")
    return parser.parse_args()


def build_micro(args: argparse.Namespace) -> MicroDuelConfig:
    terrain = Path(args.terrain) if args.terrain else None
    return replace(
        DEFAULT_CONFIG,
        map_width=float(args.map_width),
        map_height=float(args.map_height),
        episode_ticks=int(args.episode_ticks),
        units_per_team=int(args.units_per_team),
        max_entities=int(args.max_entities),
        target_slots=int(args.target_slots),
        vision_radius=float(args.vision_radius),
        terrain_path=str(terrain) if terrain is not None else None,
        map_channels=4 if terrain is not None else 3,
    )


def scripted_action(state: dict[str, Any], micro: MicroDuelConfig) -> np.ndarray:
    """Focus-fire the lowest-HP visible enemy; otherwise advance to the centre."""

    enemies = state.get("visible_enemy_entities", [])
    if not enemies:
        # Broadcast attack-move to the map centre; the game's path finding turns
        # this into a real advance instead of a straight-line teleport.
        centre_x = micro.map_width / 2.0
        centre_y = micro.map_height / 2.0
        gx = _world_to_grid(centre_x, micro.map_width, micro.position_grid)
        gy = _world_to_grid(centre_y, micro.map_height, micro.position_grid)
        return np.array([1, BROADCAST, gx, gy, 0], dtype=np.int64)
    best_index = 0
    best_key = None
    for index, enemy in enumerate(enemies):
        hp = float(enemy.get("hp") or 0.0)
        key = (hp, index)
        if best_key is None or key < best_key:
            best_key = key
            best_index = index
    return np.array([2, BROADCAST, 0, 0, best_index], dtype=np.int64)


def _world_to_grid(value: float, size: float, grid: int) -> int:
    if size <= 0 or grid <= 1:
        return grid // 2
    index = int(round(value / size * (grid - 1)))
    return max(0, min(grid - 1, index))


def run(args: argparse.Namespace) -> dict[str, Any]:
    micro = build_micro(args)
    game = RwppGameConfig(
        runtime=Path(args.runtime),
        java_home=Path(args.java_home),
        map_relative=args.map,
        map_name=args.map_name,
        episode_ticks=int(args.episode_ticks),
        map_width_px=float(args.map_width),
        map_height_px=float(args.map_height),
        units_per_team=int(args.units_per_team),
        max_entities=int(args.max_entities),
        target_slots=int(args.target_slots),
        terrain_path=Path(args.terrain) if args.terrain else None,
        win_mode=str(args.win_mode or ""),
        unlock_ai=bool(args.unlock_ai),
        frame_ms=max(0, int(args.frame_ms)),
        vision_radius=float(args.vision_radius),
    )
    results: list[dict[str, Any]] = []
    env = RwppDuelEnv(game, port=args.port_base, seed=args.seed, micro_config=micro)
    started = time.time()
    try:
        for episode in range(args.episodes):
            obs, info = env.reset(seed=args.seed + episode)
            del obs
            total = 0.0
            decisions = 0
            done = False
            final: dict[str, Any] = info
            while not done:
                state = env.state
                if args.vision_radius > 0:
                    state = apply_vision(dict(state), args.vision_radius)
                action = scripted_action(state, micro)
                _, reward, terminated, truncated, final = env.step(action)
                total += float(reward)
                decisions += 1
                done = bool(terminated or truncated)
            evaluation = final.get("eval") or {}
            record = {
                "episode": episode,
                "result": final.get("result"),
                "bridge_result": final.get("bridge_result"),
                "decisions": decisions,
                "tick": final.get("tick"),
                "return": total,
                "own_alive": final.get("own_alive"),
                "enemy_alive": final.get("enemy_alive"),
                "own_hp_frac": evaluation.get("own_hp_frac"),
                "enemy_hp_frac": evaluation.get("enemy_hp_frac"),
            }
            results.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    finally:
        env.close()
    wins = sum(1 for item in results if item["result"] == "victory")
    losses = sum(1 for item in results if item["result"] == "defeat")
    timeouts = sum(1 for item in results if item["result"] == "timeout")
    summary = {
        "episodes": len(results),
        "wins": wins,
        "losses": losses,
        "timeouts": timeouts,
        "win_rate": wins / max(1, len(results)),
        "mean_own_hp_frac": _mean(item["own_hp_frac"] for item in results),
        "mean_enemy_hp_frac": _mean(item["enemy_hp_frac"] for item in results),
        "mean_decisions": _mean(item["decisions"] for item in results),
        "total_seconds": round(time.time() - started, 2),
        "map": args.map,
        "vision_radius": args.vision_radius,
        "frame_ms": args.frame_ms,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "baseline_report.json").write_text(
        json.dumps({"summary": summary, "episodes": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def _mean(values) -> float:
    items = [float(value) for value in values if value is not None]
    return sum(items) / len(items) if items else 0.0


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
