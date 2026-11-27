"""Reproduce and explain the reserve=700 seed=71002 victory found by reserve sweep.

This is the FIRST victory under pinned engine seeds (the earlier 1/8 was an
unseeded fluke). Need to know which verdict fired (mainBuildings==0 vs
noConstructionOrTech vs dominance) and at what tick, so it can be generalised.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.probe_reserve_sweep import run

N = 6
wins = 0
for i in range(N):
    r = run(71002 + i, 28030 + i, 700.0)
    if r["res"] == "victory":
        wins += 1
    print(f"seed={71002 + i}: {r}", flush=True)
print(f"wins {wins}/{N}", flush=True)
