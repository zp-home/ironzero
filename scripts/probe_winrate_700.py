"""Reliable win-rate estimate at reserve=700, difficulty 1, Lake (pinned seeds).

Round 12 found a reproducible dominance victory at seed 71002 (bridge log:
tick 12510, ownCombat=16 vs 5, HP 3799 vs 1160) after fixing the factory-menu
death spiral. This measures a larger pinned-seed sample with both argmax and
the exact per-seed outcomes so the true rate is known before tuning further.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.probe_reserve_sweep import run

reserve = 700.0
n = 16
wins = 0
outcomes = []
for i in range(n):
    seed = 71000 + i
    r = run(seed, 28050 + i, reserve)
    outcomes.append({"seed": seed, **r})
    if r["res"] == "victory":
        wins += 1
    print(f"[{i+1}/{n}] seed={seed} {r['res']:8s} peak={r['peak']:3d} ext={r['extractors']} income={r['income']}", flush=True)

print(f"\n=== reserve={reserve} difficulty=1 Lake: {wins}/{n} = {100*wins/n:.1f}% ===", flush=True)
Path("runs/reserve700_sweep.json").write_text(
    json.dumps({"reserve": reserve, "wins": wins, "n": n, "outcomes": outcomes}, indent=2),
    encoding="utf-8",
)
