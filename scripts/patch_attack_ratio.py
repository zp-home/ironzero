"""Lower the stalled-army fallback from 60% to 35% of the cap.

With the None-cap fix applied, the attack stage now fires -- but only at step 1420
of 1600 (army 19 of cap 32, i.e. 0.6 * 32). At difficulty 1 that is too late: the
episode ends shortly after, so the army marches for the first time with almost no
time left to break any buildings. Both main bases finish at 100% HP.

The fallback exists precisely to avoid waiting for a threshold the economy cannot
fund. The measured production chain can sustain ~20 tanks while losing some to the
enemy, so 0.35 * 32 = 11 is a reachable marching size. Sending 11 tanks and
reinforcing beats sending none at all.

Usage::

    python scripts/patch_attack_ratio.py
"""

from __future__ import annotations

from pathlib import Path

PATH = Path("src/ironzero/rwpp/env.py")

OLD = "            if len(ours) < max(1, int(cap * 0.6)):\n                return []"
NEW = '''            # 0.35 of cap is reachable at difficulty 1 (~11 tanks of 32); 0.6
            # (~19) was only reached at step 1420 of 1600, too late to break a
            # building before the tick cap. March with what exists and reinforce.
            if len(ours) < max(1, int(cap * 0.35)):
                return []'''

src = PATH.read_text(encoding="utf-8")
assert src.count(OLD) == 1, f"occurrences: {src.count(OLD)}"
PATH.write_text(src.replace(OLD, NEW), encoding="utf-8", newline="\n")
print("lowered attack fallback to 35% of cap")
