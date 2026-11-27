"""Long real PPO run at difficulty 1: does the win rate actually climb?

Round 23 fixed the sampling bug (rollout_steps must cover a whole episode) and
proved the Web path reaches the engine at difficulty 1 with a pinned seed. The
remaining question the objective asks is whether *training* moves the win rate,
which has never been measured with a working loop -- every earlier "training"
run was sampling 300-tick fragments and never saw a terminal reward.

This runs a long job in the background and appends a progress line per update to
runs/train_d1_long.log so progress is visible without holding the terminal.

Config reasoning:
  * episode_ticks=1200 -> 120 decisions; rollout_steps=130 covers a whole episode
    plus slack so a terminal reward is always inside the rollout.
  * envs=4 for throughput; measured 3.8 s/update on 2 envs, so ~2 s/update here.
  * eval_every=10 with eval_episodes=8 keeps evaluation cost small but gives a
    usable CI on each checkpoint.
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

UPDATES = 120
PROGRESS = ROOT / "runs" / "train_d1_long.log"

# episode_ticks must be long enough for a decisive result to exist at all.
#
# Measured (scripts/probe_train_lengths.py, 6 pinned seeds each, difficulty 1):
#
#   ticks= 1200  decided 0/6  peak army 0   extractors 1
#   ticks= 4000  decided 0/6  peak army 3   extractors 1
#   ticks= 8000  decided 0/6  peak army 6   extractors 3
#   ticks=19674  decided 1/6  peak army 14-17  extractors 4-5
#
# At 1200 ticks the army never exists, so every episode is a timeout at a state
# where neither side can win: the only terminal signal the policy ever receives
# is ``timeout_penalty=-2.8``, which teaches nothing about winning. The long run
# at 1200 confirmed this -- 20 episodes, ``timeout_rate=1.0`` throughout.
#
# 19674 ticks is the shortest length that produces decisive results, so it is the
# shortest length that can train anything. DECISION_INTERVAL is 10, so a whole
# episode is 1968 decisions; rollout_steps=2000 covers it with slack.
EPISODE_TICKS = 19674
ROLLOUT_STEPS = 2000


def main() -> int:
    job_id = f"long-d1-{int(time.time())}"
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text("", encoding="utf-8")
    started = time.time()
    run_ppo(
        job_id,
        updates=UPDATES,
        envs=4,
        rollout_steps=ROLLOUT_STEPS,
        episode_ticks=EPISODE_TICKS,
        device="auto",
        map_name="Lake",
        map_path="maps/skirmish/[p2]Lake (2p).tmx",
        map_width=2600.0,
        map_height=2600.0,
        frame_ms=0,
        vision_radius=260.0,
        eval_every=5,
        eval_episodes=4,
        use_llm=False,
        units_per_team=None,
        max_entities=None,
        target_slots=None,
        terrain=None,
        win_mode="objective",
        unlock_ai=True,
        ai_pilot=False,
        record_ai_orders=False,
        ai_difficulty=None,
        enemy_ai_difficulty=1,
        seed=71000,
        init_checkpoint=None,
    )
    job_dir = RUNS / f"rwpp-ppo-{job_id}"
    metrics_path = job_dir / "metrics.json"
    summary_path = job_dir / "rwpp_summary.json"
    deadline = time.time() + 5400
    seen = -1
    while time.time() < deadline:
        if metrics_path.is_file():
            try:
                data = json.loads(metrics_path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                data = {}
            rows = data.get("updates") or []
            for row in rows:
                upd = int(row.get("update") or 0)
                if upd > seen:
                    seen = upd
                    line = (
                        f"upd={upd:3d} eps={row.get('episodes')} "
                        f"win={row.get('win_rate')} ret={row.get('mean_return')} "
                        f"t={time.time()-started:.0f}s"
                    )
                    print(line, flush=True)
                    with PROGRESS.open("a", encoding="utf-8") as fh:
                        fh.write(line + "\n")
            for ev in (data.get("evaluations") or []):
                upd = int(ev.get("update") or 0)
                key = f"eval-{upd}"
                if upd > 0 and key not in getattr(main, "_evals", set()):
                    if not hasattr(main, "_evals"):
                        main._evals = set()
                    main._evals.add(key)
                    line = (
                        f"EVAL upd={upd:3d} eps={ev.get('episodes')} "
                        f"win={ev.get('win_rate')} ci={ev.get('win_rate_ci95')}"
                    )
                    print(line, flush=True)
                    with PROGRESS.open("a", encoding="utf-8") as fh:
                        fh.write(line + "\n")
        if summary_path.is_file():
            break
        time.sleep(5)
    print(f"\ntotal: {time.time()-started:.0f}s", flush=True)
    if summary_path.is_file():
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        fin = s.get("final_eval") or {}
        print(f"FINAL updates={s.get('updates')} episodes={s.get('episodes')} "
              f"win_rate={fin.get('win_rate')} ci={fin.get('win_rate_ci95')}", flush=True)
        print(f"job_dir={job_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
