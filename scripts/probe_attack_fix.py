"""Test why tanks stalled at y=1037 on Lake: re-issue cadence vs grid target vs hoverTank."""
from __future__ import annotations

import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from ironzero.rwpp.config import RwppGameConfig
from ironzero.rwpp.decoding import ACTION_KINDS
from ironzero.rwpp.env import RwppDuelEnv
from ironzero.strategy.directive import MacroPlan
from scripts.probe_lake_winrate import RushStrategy


class NoAttackRush(RushStrategy):
    def propose(self, obs):
        d = super().propose(obs)
        return replace(d, macro=replace(d.macro, build_order=("extractor", "factory", "production", "expansion")))


def run_mode(mode: str, interval: int) -> None:
    wins = 0
    for i in range(8):
        seed = 71000 + i
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
        env = RwppDuelEnv(game, port=26800 + i, seed=seed, episode_prefix="atkf", strategy=NoAttackRush())
        if mode == "exact":
            # Monkey-patch grid_to_world only? No, pass exact by hooking _issue_attack_drive in a normal RushStrategy? Better: test grid interval=1 vs 10 vs 30 first!
            pass
        res = None
        com_len = 0
        avg_y = None
        try:
            env.reset(seed=seed)
            for step in range(2000):
                own = env.state.get("own_entities") or []
                foe = env.state.get("visible_enemy_entities") or []
                com = [u for u in own if not bool(u.get("is_building")) and not bool(u.get("is_builder")) and float(u.get("hp") or 0) > 0]
                com_len = len(com)
                if com:
                    avg_y = round(sum(float(u.get("y") or 0) for u in com) / len(com), 0)
                act = np.array([ACTION_KINDS.index("noop"), 0, 0, 0, 0])
                if len(com) >= 6 and step % interval == 0:
                    blds = [u for u in foe if bool(u.get("is_building"))]
                    target_u = next((u for u in blds if u.get("unit_type") == "commandCenter"), None)
                    if target_u is None and blds:
                        target_u = blds[0]
                    if target_u is None and foe:
                        target_u = foe[0]
                    if target_u is not None:
                        tx, ty = float(target_u["x"]), float(target_u["y"])
                        gx = max(0, min(7, int(tx / 2600.0 * 8)))
                        gy = max(0, min(7, int(ty / 2600.0 * 8)))
                        act = np.array([ACTION_KINDS.index("attack_move"), env.micro.groups, gx, gy, 0])
                _o, _r, t, tr, info = env.step(act)
                if t or tr:
                    res = info.get("result")
                    break
            if res == "victory":
                wins += 1
            print(f"[{mode} int={interval}] ep={i} res={res} tick={env.state.get('tick')} army={com_len} avg_y={avg_y}", flush=True)
        finally:
            env.close()
    print(f"=== {mode} int={interval}: {wins}/8 ===\n", flush=True)


def main() -> int:
    run_mode("grid_cc", 1)
    run_mode("grid_cc", 15)
    run_mode("grid_cc", 50)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
