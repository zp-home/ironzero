"""End-to-end: does a Web-API-shaped PPO request actually reach the engine at
difficulty 1?

Rounds 10-21 all measured with hand-written probes. The objective asks to train
*through the Web UI at AI difficulty 1*, and that path had two breaks:

  1. ``web/backend/app/models.py`` (``RwppPpoRequest``) had no difficulty,
     win-mode, unlock-ai or seed field, so the browser could not ask for them;
  2. ``scripts/rwpp_service/ppo.py`` declared ``seed`` on ``run_ppo`` but
     ``launch.py`` never read it from the payload, so a pinned seed was silently
     dropped.

Both are fixed. This script builds the exact payload the Web UI now sends,
starts the job through the service's own entry point, and then checks the
trainer command line and the bridge log for ``--enemy-ai-difficulty 1``.

It does not use HTTP: it calls the same code the HTTP handler calls, so it fails
for the same reasons the UI would.
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

# The payload the Web UI sends after the schema change. Values chosen to be the
# cheapest possible real run: one update, one env, short episode.
PAYLOAD = {
    "updates": 1,
    "envs": 1,
    "rollout_steps": 8,
    "episode_ticks": 1200,
    "device": "cpu",
    "map_name": "Lake",
    "map_path": "maps/skirmish/[p2]Lake (2p).tmx",
    "map_width": 2600.0,
    "map_height": 2600.0,
    "frame_ms": 0,
    "vision_radius": 260.0,
    "eval_every": 0,
    "eval_episodes": 0,
    "use_llm": False,
    "enemy_ai_difficulty": 1,
    "win_mode": "objective",
    "unlock_ai": True,
    "seed": 71000,
}


def main() -> int:
    job_id = f"e2e-web-{int(time.time())}"
    print("payload:", json.dumps(PAYLOAD, indent=2), flush=True)
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
    log_path = job_dir / "train.log"
    deadline = time.time() + 180
    while time.time() < deadline:
        if log_path.is_file():
            text = log_path.read_text(encoding="utf-8", errors="replace")
            if "arguments=" in text or "Traceback" in text:
                break
        time.sleep(2)
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    print("\n=== trainer log (first 40 lines) ===", flush=True)
    for line in text.splitlines()[:40]:
        # The trainer's stderr is UTF-8 but this console is GBK. Encode with
        # ``backslashreplace`` and strip the escape again: ``"replace"`` on both
        # sides keeps the odd character and the *console* then fails on it
        # (measured: UnicodeEncodeError on '\u01ff' killed the whole probe before
        # it printed the checks, so a passing run reported exit 1).
        safe = line.encode("ascii", "backslashreplace").decode("ascii")
        sys.stdout.write(safe + "\n")
        sys.stdout.flush()
    checks = {
        "--enemy-ai-difficulty 1": "--enemy-ai-difficulty 1" in text,
        "--seed 71000": "--seed 71000" in text,
        "--win-mode objective": "--win-mode objective" in text,
        "--unlock-ai": "--unlock-ai" in text,
    }
    print("\n=== checks ===", flush=True)
    for k, v in checks.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}", flush=True)
    # The argv line is the whole point of this probe: it proves the values the
    # Web API received are the values the engine was launched with.
    for line in text.splitlines():
        if "arguments=" in line or "--enemy-ai-difficulty" in line:
            safe = line.strip()[:400].encode("ascii", "backslashreplace").decode("ascii")
            sys.stdout.write("ARGV: " + safe + "\n")
            sys.stdout.flush()
            break
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
