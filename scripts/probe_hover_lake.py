"""Force hoverTank production (amphibious, crosses Lake water) and measure win rate.

Evidence: hoverTank is amphibious; c_tank must walk around Lake's central water,
arriving ~tick 10000 after the enemy has 4+ lightGunships. We raise hoverTank's
value-per-credit above tank so the curriculum produces it instead, then measure
8 seeds at difficulty 1 on Lake.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import ironzero.rwpp.env as env_mod

# Force hoverTank: value-per-credit above c_tank (5/350=0.0143)
env_mod._UNIT_COMBAT_RANK["hoverTank"] = 8
env_mod._UNIT_PRICE["hoverTank"] = 400

from scripts.probe_lake_winrate import run_ep  # noqa: E402


def main() -> int:
    wins = 0
    for i in range(8):
        r = run_ep("maps/skirmish/[p2]Lake (2p).tmx", "Lake", 2600.0, 2600.0,
                   71000 + i, 27000 + i, attack_at=6, max_factories=2,
                   max_extractors=4, max_upgrades=0)
        if r["res"] == "victory":
            wins += 1
        print(f"HOVER ep={i}: {r}", flush=True)
    print(f"HOVER total: {wins}/8", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
