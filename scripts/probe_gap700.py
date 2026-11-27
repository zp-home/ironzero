"""At reserve=700 (the config that proved dominance is reachable), measure the
end-of-episode own vs enemy COMBAT counts, to quantify the exact gap.

Dominance needs: ownCombat >= 8 AND ownCombat >= 3*enemyCombat AND
ownCombatHp >= 3*enemyHp. The one win had 16 vs 5 / 3799 vs 1160. If losses end
near e.g. 20 vs 10, the gap is "kill 7 more enemy units", not "grow the army".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from collections import Counter
from ironzero.rwpp.config import RwppGameConfig
from ironzero.rwpp.env import RwppDuelEnv
from scripts.probe_lake_winrate import RushStrategy


def combat(entities) -> list:
    return [u for u in entities
            if u.get("can_attack") and not u.get("is_builder")
            and not u.get("is_building") and float(u.get("hp") or 0) > 0]


def run(seed: int, port: int) -> dict:
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
        economy_army_cap=40,
        economy_army_target=24,
        economy_max_extractors=6,
        economy_max_factories=2,
        economy_max_turrets=0,
        economy_max_upgrades=0,
        economy_credit_reserve=700.0,
        economy_production_queue_depth=4,
    )
    env = RwppDuelEnv(game, port=port, seed=seed, episode_prefix="gap700",
                      strategy=RushStrategy(attack_at=8, max_factories=2,
                                            max_extractors=6, reserve=700.0,
                                            army_cap=40))
    own_peak = foe_peak = 0
    res = None
    end_own = end_foe = None
    end_own_hp = end_foe_hp = 0.0
    try:
        env.reset(seed=seed)
        for _ in range(19674 // 10 + 60):
            _o, _r, t, tr, info = env.step(np.array([9, 0, 0, 0, 0]))
            oc = combat(env.state.get("own_entities") or [])
            fc = combat(env.state.get("visible_enemy_entities") or [])
            own_peak = max(own_peak, len(oc))
            foe_peak = max(foe_peak, len(fc))
            if t or tr:
                res = info.get("result")
                end_own, end_foe = oc, fc
                end_own_hp = sum(float(u.get("hp") or 0) for u in oc)
                end_foe_hp = sum(float(u.get("hp") or 0) for u in fc)
                break
    finally:
        env.close()
    return {
        "res": res,
        "own_peak": own_peak, "foe_peak": foe_peak,
        "end_own": len(end_own or []), "end_foe": len(end_foe or []),
        "end_own_hp": round(end_own_hp), "end_foe_hp": round(end_foe_hp),
        "own_types": dict(Counter(str(u.get("unit_type")) for u in (end_own or []))),
        "foe_types": dict(Counter(str(u.get("unit_type")) for u in (end_foe or []))),
    }


def main() -> int:
    port = 28160
    rows = []
    for i in range(8):
        seed = 71000 + i
        r = run(seed, port)
        port += 1
        rows.append({"seed": seed, **r})
        print(f"seed={seed} {r['res']:8s} peak own/foe={r['own_peak']}/{r['foe_peak']} "
              f"end={r['end_own']}/{r['end_foe']} hp={r['end_own_hp']}/{r['end_foe_hp']}", flush=True)
        if r["res"] != "victory":
            print(f"    own={r['own_types']}", flush=True)
            print(f"    foe={r['foe_types']}", flush=True)
    wins = sum(1 for r in rows if r["res"] == "victory")
    print(f"\nwins {wins}/{len(rows)}", flush=True)
    Path("runs/gap700.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
