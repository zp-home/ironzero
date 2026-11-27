"""Compare seed 71000 (timeout) vs 71001 (victory @ 4747) & inspect raw factory/unit fields."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from ironzero.rwpp.config import RwppGameConfig
from ironzero.rwpp.env import RwppDuelEnv
from scripts.probe_lake_winrate import RushStrategy


def trace(seed: int, port: int) -> None:
    print(f"\n=== TRACE seed={seed} ===", flush=True)
    game = RwppGameConfig(
        map_relative="maps/skirmish/[p2]Lake (2p).tmx",
        map_name="Lake",
        map_width_px=2600.0,
        map_height_px=2600.0,
        episode_ticks=19674,
        enemy_ai_difficulty=1,
        unlock_ai=True,
        win_mode="objective",
        frame_ms=0,
        physics_ms=16,
        lockstep=True,
        max_entities=16,
        units_per_team=16,
        economy_army_cap=32,
        economy_army_target=24,
        economy_max_extractors=4,
        economy_max_factories=2,
        economy_max_turrets=0,
        economy_max_upgrades=0,
        economy_credit_reserve=350.0,
        economy_production_queue_depth=4,
    )
    env = RwppDuelEnv(
        game, port=port, seed=seed, episode_prefix="bdiag",
        strategy=RushStrategy(attack_at=6, max_factories=2, max_extractors=4, max_upgrades=0),
    )
    try:
        env.reset(seed=seed)
        dumped_factory = False
        for step in range(2000):
            _o, _r, t, tr, info = env.step(np.array([9, 0, 0, 0, 0]))
            co = getattr(env, "curriculum_order", None)
            if step in (150, 250, 350, 400, 450, 500, 600, 750, 900, 1200, 1500, 1800) or t or tr or (co and co.get("kind") in ("attack_move", "move_group") and step % 50 == 0):
                own = env.state.get("own_entities") or []
                foe = env.state.get("visible_enemy_entities") or []
                com = [u for u in own if not bool(u.get("is_building")) and not bool(u.get("is_builder"))]
                avg_xy = (
                    round(sum(float(u.get("x", 0)) for u in com) / max(1, len(com)), 0),
                    round(sum(float(u.get("y", 0)) for u in com) / max(1, len(com)), 0),
                ) if com else None
                cc = next((u for u in own if u.get("unit_type") == "commandCenter"), None)
                ecc = [u for u in foe if u.get("unit_type") == "commandCenter"]
                ecc_info = [(round(u["x"], 0), round(u["y"], 0), u.get("hp"), u.get("max_hp")) for u in ecc]
                own_xy = (round(cc["x"], 0), round(cc["y"], 0)) if cc else None
                print(
                    f"step={step:4d} tick={env.state.get('tick') or info.get('tick')} own_cc={own_xy} army={len(com)} avg={avg_xy} "
                    f"foe_cc={ecc_info} foe={dict(Counter(u.get('unit_type') for u in foe))} "
                    f"cred={(env.state.get('global_state') or {}).get('funds')} macro_atk={env.macro.attack_at_army} macro_ord={env.macro.stage_names} co={co}",
                    flush=True,
                )
                if not dumped_factory and (env.state.get("factories") or []):
                    for f in env.state.get("factories") or []:
                        if f.get("unit_type") != "commandCenter":
                            print("  RAW FACTORY:", f, flush=True)
                            dumped_factory = True
                            break
            if t or tr:
                print(f"RESULT={info.get('result')} at tick={env.state.get('tick')}", flush=True)
                break
    finally:
        env.close()


def main() -> int:
    trace(71001, 26711)
    trace(71000, 26710)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
