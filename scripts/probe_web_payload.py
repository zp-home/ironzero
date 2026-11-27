"""Does the Web UI payload now actually carry difficulty 1, win mode and seed?

The objective says "train through the Web UI at AI difficulty 1". Rounds 23-24
proved the *host service* accepts those fields and forwards them to the trainer,
but the browser never sent them: ``web/frontend/js/ppoRun.js`` built a payload
with only the training hyper-parameters, so every UI-started run played at the
engine default (level 0, x1.00 income) regardless of what the form showed.

This probe verifies the fixed chain end to end at the layer that can actually be
checked without a browser:

  1. the Pydantic schema accepts the new fields (``web/backend/app/models.py``);
  2. the host service parses them out of the payload (``launch.py``);
  3. the trainer command line the service builds contains the matching flags.

It does not launch a game, so it runs in seconds and is safe in CI.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# The payload shape ``feStartPpo`` now builds. Keep it in sync with
# ``web/frontend/js/ppoRun.js``: this dict is the contract between the browser
# and every layer below it.
WEB_PAYLOAD = {
    "updates": 1,
    "envs": 1,
    "rollout_steps": 8,
    "episode_ticks": 19674,
    "device": "cpu",
    "map_name": "Lake",
    "map_path": "maps/skirmish/[p2]Lake (2p).tmx",
    "map_width": 2600.0,
    "map_height": 2600.0,
    "frame_ms": 0,
    "vision_radius": 260.0,
    "eval_every": 0,
    "eval_episodes": 1,
    "use_llm": False,
    "units_per_team": 2,
    "init_mode": "scratch",
    # The four fields this round added to the frontend payload.
    "enemy_ai_difficulty": 1,
    "win_mode": "objective",
    "unlock_ai": True,
    "seed": 71000,
}


def check_schema() -> bool:
    """The backend model must accept the fields the browser now sends."""

    # Loaded by path on purpose. ``web/backend/app/models.py`` imports nothing
    # but pydantic, so importing it as part of its package would drag in the
    # whole FastAPI app (and fail outside the container) for no reason.
    import importlib.util

    model_path = ROOT / "web" / "backend" / "app" / "models.py"
    if not model_path.is_file():  # pragma: no cover - layout guard
        print(f"  SKIP  {model_path} not found")
        return True
    spec = importlib.util.spec_from_file_location("_ironzero_web_models", model_path)
    if spec is None or spec.loader is None:  # pragma: no cover - layout guard
        print("  SKIP  could not build an import spec for models.py")
        return True
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"  SKIP  models.py import failed ({type(exc).__name__}: {exc})")
        return True
    request_model = getattr(module, "RwppPpoRequest", None)
    if request_model is None:  # pragma: no cover - layout guard
        print("  SKIP  RwppPpoRequest not found in models.py")
        return True
    model = request_model(**WEB_PAYLOAD)
    ok = (
        model.enemy_ai_difficulty == 1
        and model.win_mode == "objective"
        and model.unlock_ai is True
        and model.seed == 71000
    )
    print(f"  {'OK  ' if ok else 'FAIL'} schema: enemy_ai_difficulty="
          f"{model.enemy_ai_difficulty} win_mode={model.win_mode!r} "
          f"unlock_ai={model.unlock_ai} seed={model.seed}")
    return ok


def check_command_line() -> bool:
    """The service must translate the payload into the trainer's flags."""

    from rwpp_service.launch import _optional_int
    from rwpp_service.ppo import resolve_preset

    enemy = _optional_int(WEB_PAYLOAD.get("enemy_ai_difficulty"), None)
    seed = _optional_int(WEB_PAYLOAD.get("seed"), None)
    preset = resolve_preset(
        WEB_PAYLOAD, WEB_PAYLOAD["map_name"], WEB_PAYLOAD["map_path"]
    )
    flags: list[str] = []
    if enemy is not None:
        flags.extend(["--enemy-ai-difficulty", str(enemy)])
    if preset["win_mode"]:
        flags.extend(["--win-mode", preset["win_mode"]])
    if preset["unlock_ai"]:
        flags.append("--unlock-ai")
    if seed is not None:
        flags.extend(["--seed", str(seed)])
    expected = ["--enemy-ai-difficulty", "1", "--win-mode", "objective", "--unlock-ai"]
    ok = all(item in flags for item in expected) and "--seed" in flags
    print(f"  {'OK  ' if ok else 'FAIL'} flags: {' '.join(flags)}")
    return ok


def check_frontend_source() -> bool:
    """The browser source must mention every field it claims to send."""

    source = (ROOT / "web" / "frontend" / "js" / "ppoRun.js").read_text(
        encoding="utf-8", errors="replace"
    )
    required = ("enemy_ai_difficulty", "win_mode", "unlock_ai", "seed")
    missing = [name for name in required if name not in source]
    ok = not missing
    print(f"  {'OK  ' if ok else 'FAIL'} ppoRun.js sends: "
          f"{', '.join(required)}"
          + (f" (missing {missing})" if missing else ""))
    return ok


def main() -> int:
    print("Web PPO payload contract")
    print("  payload:", json.dumps(
        {k: WEB_PAYLOAD[k] for k in
         ("enemy_ai_difficulty", "win_mode", "unlock_ai", "seed")}
    ))
    results = [check_schema(), check_command_line(), check_frontend_source()]
    passed = sum(1 for r in results if r)
    print(f"\n{passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
