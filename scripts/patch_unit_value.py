"""Rank producible units by combat value per credit, not by absolute value.

The previous patch preferred the strongest unit in the menu, which on a land factory
is ``artillery`` (rank 5, 900 credits). Measured consequence over a full episode:
the army ended as an *empty set* while income reached 42 and three land factories
stood ready -- the factory was asked for something the bank could never pay for, so
it produced nothing at all. Six scouts (the state before that patch) at least fought.

The right ordering is value per credit, so a 350-credit tank (rank 4) beats a
900-credit artillery (rank 5) whenever the economy is small, and the artillery only
wins when the bank is large enough to buy it without stalling production.

This module is imported by the env; keeping the table here makes the ordering
inspectable and testable on its own.

Usage::

    python scripts/patch_unit_value.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "src" / "ironzero" / "rwpp" / "env.py"

OLD = '''#: Rough combat value per unit type, used to prefer a real fighting unit over the
#: cheapest one. Higher is better. Unknown types rank 0.
#:
#: The numbers are only an ordering, not a balance claim: what matters is that
#: ``scout`` ranks below every actual combat unit, because a menu-driven
#: "first entry" policy was measured to build nothing but scouts while three land
#: factories stood idle (``.runtime`` endgame probe on Lake at difficulty 1).
_UNIT_COMBAT_RANK: dict[str, int] = {
    "scout": 1,
    "hoverTank": 3,
    "tank": 4,
    "c_tank": 5,
    "artillery": 5,
    "c_artillery": 6,
    "lightGunship": 4,
    "c_helicopter": 5,
    "gunship": 5,
    "bomber": 6,
    "c_bomber": 6,
}


def _unit_combat_rank(name: str) -> int:
    """Combat value of a producible unit name; 0 when unknown."""
    return _UNIT_COMBAT_RANK.get(name.split(".")[-1], 0)'''

NEW = '''#: Rough combat value per unit type. Higher is better; unknown types rank 0.
#:
#: These are an *ordering*, not a balance claim. What matters is that ``scout`` sits
#: below every real combat unit, because a menu-driven "first entry" policy was
#: measured to build nothing but scouts while three land factories stood idle
#: (endgame probe on Lake at difficulty 1).
_UNIT_COMBAT_RANK: dict[str, int] = {
    "scout": 1,
    "hoverTank": 3,
    "tank": 4,
    "c_tank": 5,
    "artillery": 5,
    "c_artillery": 6,
    "lightGunship": 4,
    "c_helicopter": 5,
    "gunship": 5,
    "bomber": 6,
    "c_bomber": 6,
}

#: Rough price per unit type, in credits. Used only to compare value per credit.
_UNIT_PRICE: dict[str, int] = {
    "scout": 700,
    "hoverTank": 450,
    "tank": 350,
    "c_tank": 500,
    "artillery": 900,
    "c_artillery": 1000,
    "lightGunship": 700,
    "c_helicopter": 900,
    "gunship": 900,
    "bomber": 1000,
    "c_bomber": 1200,
}


def _unit_combat_rank(name: str) -> int:
    """Combat value of a producible unit name; 0 when unknown."""
    return _UNIT_COMBAT_RANK.get(name.split(".")[-1], 0)


def _unit_value_per_credit(name: str) -> float:
    """Combat value per credit, so an affordable unit beats a better unaffordable one.

    Ordering by absolute strength alone was measured to *empty* the army: the
    preferred unit on a land factory is ``artillery`` at 900 credits, the factory
    waited for a bank it never saw, and the episode ended with zero mobile units
    while income sat at 42 and three factories were idle. A 350-credit tank that is
    actually built is worth more than a 900-credit artillery that is not.
    """
    key = name.split(".")[-1]
    rank = _UNIT_COMBAT_RANK.get(key, 0)
    if rank <= 0:
        return 0.0
    price = max(1, _UNIT_PRICE.get(key, 500))
    return rank / float(price)'''

# The selection loop must compare value-per-credit, not raw rank.
OLD_LOOP = '''        target_index = None
        best_rank = None
        for index, (factory_id, unit_name) in enumerate(options):
            if factory_id not in combat_ids:
                continue
            rank = _unit_combat_rank(str(unit_name))
            if best_rank is None or rank > best_rank:
                best_rank = rank
                target_index = index'''

NEW_LOOP = '''        target_index = None
        best_score = None
        for index, (factory_id, unit_name) in enumerate(options):
            if factory_id not in combat_ids:
                continue
            # Value per credit: a tank that gets built beats an artillery that the
            # bank can never afford. See ``_unit_value_per_credit``.
            score = _unit_value_per_credit(str(unit_name))
            if best_score is None or score > best_score:
                best_score = score
                target_index = index'''

OLD_DOC = '''        # Prefer the *strongest* combat unit a real factory can make, not the first
        # entry in the menu.'''

NEW_DOC = '''        # Prefer the best *value per credit* a real factory can make, not the first
        # entry in the menu and not the single strongest unit.'''


def main() -> int:
    src = PATH.read_text(encoding="utf-8")

    if "_unit_value_per_credit" in src:
        print("already patched")
        return 0

    for old, new, label in (
        (OLD, NEW, "rank table"),
        (OLD_LOOP, NEW_LOOP, "selection loop"),
        (OLD_DOC, NEW_DOC, "comment"),
    ):
        assert src.count(old) == 1, f"{label} occurrences: {src.count(old)}"
        src = src.replace(old, new)

    PATH.write_text(src, encoding="utf-8", newline="\n")
    print("patched unit selection to value-per-credit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
