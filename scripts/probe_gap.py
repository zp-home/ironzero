"""Measure end-state army balance at difficulty 1 to quantify the gap."""
from __future__ import annotations

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


def main() -> int:
    for seed in (71000, 71001, 71002):
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
            max_entities=8,
            units_per_team=8,
            economy_army_cap=32,
            economy_army_target=24,
            economy_max_extractors=4,
            economy_max_factories=2,
            economy_max_turrets=0,
            economy_max_upgrades=0,
            economy_credit_reserve=350.0,
            economy_production_queue_depth=4,
        )
        env = RwppDuelEnv(game, port=27600 + seed % 100, seed=seed, episode_prefix="gap",
                          strategy=RushStrategy(attack_at=6, max_factories=2, max_extractors=4))
        try:
            env.reset(seed=seed)
            peak_own = peak_foe = 0
            for _ in range(19674 // 10 + 50):
                _o, _r, t, tr, info = env.step(np.array([9, 0, 0, 0, 0]))
                own = env.state.get("own_entities") or []
                foe = env.state.get("visible_enemy_entities") or []
                oc = sum(1 for u in own if u.get("can_attack") and not u.get("is_builder") and not u.get("is_building"))
                fc = sum(1 for u in foe if u.get("can_attack") and not u.get("is_builder") and not u.get("is_building"))
                peak_own = max(peak_own, oc)
                peak_foe = max(peak_foe, fc)
                if t or tr:
                    break
            print(f"seed={seed} peak_own={peak_own} peak_foe={peak_foe}", flush=True)
            print(f"  own_end={dict(Counter(str(u.get('unit_type')) for u in (env.state.get('own_entities') or [])))}", flush=True)
            print(f"  foe_end={dict(Counter(str(u.get('unit_type')) for u in (env.state.get('visible_enemy_entities') or [])))}", flush=True)
        finally:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
