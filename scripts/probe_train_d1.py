"""Does real PPO training through the Web path move the difficulty-1 win rate?

Rounds 10-22 measured hand-written curricula. The objective is about *training*
through the Web UI at difficulty 1, and round 23 proved that path now works
end to end (difficulty 1 and the seed reach the engine).

This runs an actual PPO job through the service entry point -- the same code the
browser's /api/rwpp/ppo hits -- with the real game at difficulty 1, then reads
the metrics the trainer wrote. It answers two questions the goal cares about:

  * is training fast enough to iterate (wall clock per update)?
  * does the win rate move at all from a scratch policy?

Deliberately small (2 envs, 30 rollout steps, short episodes) so it finishes in
minutes rather than hours; the point is the shape of the learning curve, not a
final number.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rwpp_service.paths import RUNS
from rwpp_service.ppo import run_ppo

UPDATES = 8
PAYLOAD = {
    "updates": UPDATES,
    "envs": 2,
    # rollout_steps must cover a whole episode or PPO never sees a terminal
    # reward. DECISION_INTERVAL is 10 ticks, so episode_ticks=1200 needs 120
    # decisions; 30 steps (300 ticks) ended every episode mid-game and the
    # metrics showed episodes=0 for 5 of 6 updates -- the trainer was sampling
    # 300-tick fragments and never learning from a win or a loss.
    "rollout_steps": 130,
    "episode_ticks": 1200,
    "device": "auto",
    "map_name": "Lake",
    "map_path": "maps/skirmish/[p2]Lake (2p).tmx",
    "map_width": 2600.0,
    "map_height": 2600.0,
    "frame_ms": 0,
    "vision_radius": 260.0,
    "eval_every": 2,
    "eval_episodes": 3,
    "use_llm": False,
    "enemy_ai_difficulty": 1,
    "win_mode": "objective",
    "unlock_ai": True,
    "seed": 71000,
}


def main() -> int:
    job_id = f"train-d1-{int(time.time())}"
    started = time.time()
    run_ppo(
        job_id,
        updates=PAYLOAD["updates"],
        envs=PAYLOAD["envs"],
        rollout_steps=PAYLOAD["rollout_steps"],
        episode_ticks=PAYLOAD["episode_ticks"],
        device=PAYLOAD["device"],
        map_name=PAYLOAD["map_name"],
        map_path=PAYLOAD["map_path"],
        map_width=PAYLOAD["map_width"],
        map_height=PAYLOAD["map_height"],
        frame_ms=PAYLOAD["frame_ms"],
        vision_radius=PAYLOAD["vision_radius"],
        eval_every=PAYLOAD["eval_every"],
        eval_episodes=PAYLOAD["eval_episodes"],
        use_llm=PAYLOAD["use_llm"],
        units_per_team=None,
        max_entities=None,
        target_slots=None,
        terrain=None,
        win_mode=PAYLOAD["win_mode"],
        unlock_ai=PAYLOAD["unlock_ai"],
        ai_pilot=False,
        record_ai_orders=False,
        ai_difficulty=None,
        enemy_ai_difficulty=PAYLOAD["enemy_ai_difficulty"],
        seed=PAYLOAD["seed"],
        init_checkpoint=None,
    )
    job_dir = RUNS / f"rwpp-ppo-{job_id}"
    metrics_path = job_dir / "metrics.json"
    summary_path = job_dir / "rwpp_summary.json"
    deadline = time.time() + 1800
    last = None
    while time.time() < deadline:
        if summary_path.is_file():
            break
        if metrics_path.is_file():
            try:
                data = json.loads(metrics_path.read_text(encoding="utf-8"))
                rows = data.get("updates") or data.get("history") or []
                if isinstance(rows, list) and rows:
                    latest = rows[-1]
                    tag = json.dumps(latest, sort_keys=True)
                    if tag != last:
                        last = tag
                        wr = latest.get("eval_win_rate", latest.get("win_rate"))
                        print(f"  update {latest.get('update')} eval_win={wr} "
                              f"elapsed={time.time()-started:.0f}s", flush=True)
            except (ValueError, OSError):
                pass
        time.sleep(5)
    elapsed = time.time() - started
    print(f"\ntotal wall clock: {elapsed:.0f}s for {UPDATES} updates", flush=True)
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        final = summary.get("final_eval") or {}
        print(f"final: updates={summary.get('updates')} episodes={summary.get('episodes')} "
              f"win_rate={final.get('win_rate')} "
              f"ci={final.get('win_rate_ci95')} "
              f"mean_return={final.get('mean_return')}", flush=True)
        print(f"report: {final.get('report_path')}", flush=True)
    else:
        print("no summary written (job may still be running)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
