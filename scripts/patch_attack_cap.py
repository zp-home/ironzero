"""Fix the attack stage's 60%-cap fallback reading a None army cap.

Decisive measurement (scripts/probe_attack_stage_gate.py): over 1600 decisions the
attack stage fired **zero** orders; every sample read ``army<20``. The stage's own
gate has a fallback meant to handle exactly this ("a stalled army must still
march"):

    cap = max(int(self.macro.army_cap), int(self.macro.attack_at_army))
    if len(ours) < max(1, int(cap * 0.6)):
        return []

But the rule layer leaves ``macro.army_cap = None`` (the CLI value lives on the
game config). ``int(None)`` raises TypeError, which the dispatcher swallows, so the
stage silently returns nothing. From the outside that is indistinguishable from
"the army never got big enough" -- with 15 tanks sitting 1250 px from an enemy
building and both main bases at 100% HP.

Fix: read the cap from the authoritative game config, falling back to the macro
value and finally to the attack threshold itself. Never call int() on None.

Usage::

    python scripts/patch_attack_cap.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "src" / "ironzero" / "rwpp" / "env.py"

OLD = '''            cap = max(int(self.macro.army_cap), int(self.macro.attack_at_army))
            if len(ours) < max(1, int(cap * 0.6)):
                return []'''

NEW = '''            # The cap comes from the game config, not ``macro.army_cap``: the rule
            # layer leaves that field unset (None), and ``int(None)`` raised
            # TypeError on every decision, which the dispatcher swallowed as "the
            # stage had nothing to say". Measured on Lake at difficulty 1: the
            # attack stage fired zero orders in 1600 decisions while 15 tanks sat
            # 1250 px from an enemy building and both bases stayed at 100% HP.
            macro_cap = getattr(self.macro, "army_cap", None)
            game_cap = getattr(self.game, "economy_army_cap", None)
            cap = int(macro_cap) if macro_cap is not None else (
                int(game_cap) if game_cap is not None else int(self.macro.attack_at_army)
            )
            cap = max(cap, int(self.macro.attack_at_army))
            if len(ours) < max(1, int(cap * 0.6)):
                return []'''


def main() -> int:
    src = PATH.read_text(encoding="utf-8")

    if "macro_cap = getattr(self.macro" in src:
        print("already patched")
        return 0

    assert src.count(OLD) == 1, f"occurrences: {src.count(OLD)}"
    PATH.write_text(src.replace(OLD, NEW), encoding="utf-8", newline="\n")
    print("patched attack cap fallback")
    return 0


if __name__ == "__main__":
    sys.exit(main())
