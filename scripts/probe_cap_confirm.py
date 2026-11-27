"""Confirm army_cap=56 with a 16-seed sample, and probe higher caps.

The sweep found reserve=700/cap=56 at 3/8 (37.5%) vs 0/8 at cap=40. Dominance
needs ownCombat >= 3*enemyCombat, so a cap that lets the army keep growing past
the army_target (24) is exactly what converts near-misses into wins.

Caveat to control for: samples of 8 swing wildly (round 13 saw 4/16 then 3/16 on
identical config). This runs 16 seeds per cell so the comparison is meaningful.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.probe_sweep14 import run

cells = [(700.0, 56), (700.0, 72), (700.0, 40)]
n = 16
port = 28350
report = []
for reserve, cap in cells:
    rows = []
    wins = 0
    for i in range(n):
        seed = 71000 + i
        r = run(seed, port, reserve, cap)
        port += 1
        rows.append({"seed": seed, **r})
        if r["res"] == "victory":
            wins += 1
    peaks = [r["peak"] for r in rows]
    report.append({"reserve": reserve, "army_cap": cap, "wins": wins, "n": n, "rows": rows})
    print(f"reserve={reserve:5.0f} cap={cap:3d}  wins={wins}/{n} = {100*wins/n:.1f}%  "
          f"peak={min(peaks)}-{max(peaks)}", flush=True)
Path("runs/cap56_confirm.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
