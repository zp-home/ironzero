# SPDX-License-Identifier: Apache-2.0
"""Report a checkpoint's win rate under **both** argmax and sampling.

Why this exists
---------------

The training loop's own evaluation runs the policy with ``deterministic=True``
(argmax per head), while the rollouts that produced those weights sampled from the
same heads. Those two policies are not the same object, and the gap is not small:
measured on one 20-update checkpoint on Lake at difficulty -2, with identical seeds,

    deterministic=True   (argmax)   win 0.000   own_alive 8,8,8,8,1,1,1,2
    deterministic=False  (sample)   win 0.375   own_alive 10,17,19,19,19,11,10,19

The mechanism is visible in the action distribution: the argmax policy emitted
``move_group`` on **298 of 300** decisions, and a move order re-tasks the builder, so
every one of them restarts an in-flight construction. The sampled policy spread its
orders over nine kinds and its army grew. Reporting only the argmax number would
have described that checkpoint as "0% -- cannot win", which is not what it does when
it is actually played.

So: a single win rate is not a measurement of a policy, it is a measurement of a
*policy plus a decoding rule*. This probe reports the pair, and the 95% Wilson
interval for each, so a comparison cannot be won or lost by choosing the flattering
decoding.

Usage::

    python scripts/eval_two_modes.py <checkpoint> [--episodes 16] [--difficulty 1]

Writes ``<checkpoint dir>/eval_two_modes.json`` and prints a table.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def wilson(wins: int, episodes: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval; the sample sizes here are far too small for a
    normal approximation to be honest (16 games cannot distinguish 0.6 from 0.9)."""

    if episodes <= 0:
        return (0.0, 1.0)
    p = wins / episodes
    denominator = 1.0 + z * z / episodes
    centre = (p + z * z / (2 * episodes)) / denominator
    half = z * math.sqrt(p * (1 - p) / episodes + z * z / (4 * episodes * episodes)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--difficulty", type=int, default=1)
    ap.add_argument("--map", default="maps/skirmish/[p2]Lake (2p).tmx")
    ap.add_argument("--map-name", default="Lake")
    ap.add_argument("--map-width", type=float, default=2600.0)
    ap.add_argument("--map-height", type=float, default=2600.0)
    ap.add_argument("--episode-ticks", type=int, default=19674)
    ap.add_argument("--port-base", type=int, default=17_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=90_000)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from ironzero.rwpp.config import RwppGameConfig
    from ironzero.training.evaluate_rwpp import evaluate_rwpp_policy
    from ironzero.training.ppo import load_policy

    policy, micro = load_policy(args.checkpoint, device=args.device)
    game = RwppGameConfig(
        map_relative=args.map,
        map_name=args.map_name,
        map_width_px=args.map_width,
        map_height_px=args.map_height,
        episode_ticks=args.episode_ticks,
        enemy_ai_difficulty=args.difficulty,
        unlock_ai=True,
        win_mode="objective",
        frame_ms=1,
        physics_ms=16,
        max_entities=int(micro.max_entities),
        units_per_team=int(micro.units_per_team),
    )

    report: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "difficulty": args.difficulty,
        "map": args.map_name,
        "episodes": args.episodes,
        "obs_dim": int(micro.flat_obs_dim),
        "modes": {},
    }
    print(
        f"checkpoint={args.checkpoint}\n"
        f"difficulty={args.difficulty} map={args.map_name} episodes={args.episodes} "
        f"obs_dim={micro.flat_obs_dim}"
    )
    for index, (name, deterministic) in enumerate(
        (("argmax", True), ("sampled", False))
    ):
        result = evaluate_rwpp_policy(
            policy,
            game=game,
            episodes=args.episodes,
            micro_config=micro,
            port_base=args.port_base + index * 200,
            seed=args.seed,
            device=args.device,
            deterministic=deterministic,
            parallel=args.parallel,
        )
        wins = int(result.get("wins", 0) or 0)
        episodes = int(result.get("episodes", args.episodes) or args.episodes)
        low, high = wilson(wins, episodes)
        own = [ep.get("own_alive") for ep in result.get("episode_results", [])]
        enemy = [ep.get("enemy_alive") for ep in result.get("episode_results", [])]
        report["modes"][name] = {
            "deterministic": deterministic,
            "wins": wins,
            "losses": int(result.get("losses", 0) or 0),
            "timeouts": int(result.get("timeouts", 0) or 0),
            "episodes": episodes,
            "win_rate": wins / episodes if episodes else 0.0,
            "win_rate_ci95": [round(low, 4), round(high, 4)],
            "clears_80pct_lower_bound": low >= 0.80,
            "own_alive": own,
            "enemy_alive": enemy,
            "mean_own_hp_frac": result.get("mean_own_hp_frac"),
            "mean_decisions": result.get("mean_decisions"),
        }
        print(
            f"  {name:<8} win={wins}/{episodes} = {wins / max(1, episodes):.3f}  "
            f"95% CI [{low:.3f}, {high:.3f}]  own={own}"
        )

    out_path = Path(args.out) if args.out else Path(args.checkpoint).parent / "eval_two_modes.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    best = max(
        report["modes"].values(), key=lambda m: m["win_rate"]
    )
    print(
        f"best mode win_rate={best['win_rate']:.3f} "
        f"clears 80% at 95% confidence: {best['clears_80pct_lower_bound']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
