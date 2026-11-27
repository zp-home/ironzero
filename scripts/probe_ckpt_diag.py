"""Inspect what d1_attack_fixed policy actually outputs and how the match unfolds."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch
from ironzero.rwpp.config import RwppGameConfig
from ironzero.rwpp.env import RwppDuelEnv
from ironzero.rwpp.decoding import ACTION_KINDS
from ironzero.training.policy import masks_to_tensors
from ironzero.training.ppo import load_policy
from ironzero.strategy.rule_strategy import RuleStrategy


def main() -> int:
    ckpt = Path(r"E:\qwq\强化学习笔记\铁锈战争\runs\platform\d1_attack_fixed\checkpoint_best.pt")
    policy, micro = load_policy(ckpt, device="cpu")
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
        max_entities=int(micro.max_entities),
        units_per_team=int(micro.units_per_team),
        economy_army_cap=32,
        economy_army_target=20,
        economy_max_extractors=9,
        economy_max_factories=3,
        economy_production_queue_depth=3,
    )
    env = RwppDuelEnv(
        game, port=26_100, seed=70000, micro_config=micro,
        episode_prefix="ckptdiag", strategy=RuleStrategy(),
    )
    action_counts = Counter()
    group_counts = Counter()
    curriculum_counts = Counter()
    own_peak = 0
    try:
        obs, info = env.reset(seed=70000)
        foes0 = env.state.get("visible_enemy_entities") or []
        print("initial visible foes:", [(f.get("unit_type"), f.get("x"), f.get("y")) for f in foes0])
        for step in range(2100):
            mask_tensors = masks_to_tensors([info["masks"]], torch.device("cpu"))
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                actions, _, _ = policy.act(obs_t, mask_tensors, deterministic=True)
            act = actions[0].cpu().numpy()
            action_counts[ACTION_KINDS[int(act[0])]] += 1
            group_counts[int(act[1])] += 1
            obs, _r, term, trunc, info = env.step(act)
            co = getattr(env, "curriculum_order", None)
            if isinstance(co, dict) and co.get("kind"):
                curriculum_counts[co["kind"]] += 1
            own = env.state.get("own_entities") or []
            own_peak = max(
                own_peak,
                sum(1 for u in own if u.get("can_attack") and not u.get("is_builder") and not u.get("is_building")),
            )
            if step in (300, 600, 900, 1200, 1500, 1800) or term or trunc:
                g = env.state.get("global_state") or {}
                print(
                    f"step={step} tick={env.state.get('tick')} income={g.get('income')} "
                    f"funds={round(float(g.get('funds') or 0))} own_alive={g.get('own_alive')} "
                    f"enemy_alive={g.get('enemy_alive')} own_hp={g.get('own_hp')} enemy_hp={g.get('enemy_hp')}"
                )
            if term or trunc:
                print("DONE:", info.get("result"))
                break
        print("policy argmax actions:", dict(action_counts))
        print("policy argmax groups:", dict(group_counts))
        print("curriculum orders:", dict(curriculum_counts))
        print("own peak combat:", own_peak)
    finally:
        try:
            env.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
