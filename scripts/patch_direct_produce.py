"""Emit production orders as concrete (factory, unit) pairs instead of an index.

Why the index had to go
-----------------------
``produce_unit`` was encoded as a target-head index into ``produce_options(state)``,
a list rebuilt from the live observation. That list is **not stable between
decisions**: a busy or under-construction factory reports a shorter ``can_produce``
menu, so the same index names a different (factory, unit) pair on the next decision.
Measured with scripts/probe_produce_options.py on Lake at difficulty 1:

    step 83: produce_options = 13 entries
             [[23,scout],[27,hoverTank],[27,tank],[27,artillery],[27,scout],
              [28,hoverTank],...]
    step 85: produce_options =  6 entries      (three factories busy)
             [[27,hoverTank],[27,tank],[28,hoverTank],[28,tank],[30,hoverTank],
              [30,tank]]

and the queue held ``c_tank`` with progress -1.0 for the rest of the episode: a unit
the land factory cannot build, queued forever. Because ``production_queue_depth``
counts that wedged queue, the stage then read ``queue_full`` on 771 of 900 decisions
and the army stayed at **zero** while income reached 42 and three factories stood
idle.

The bridge already accepts the concrete form
-------------------------------------------
``IronZeroBridge`` handles ``produce_unit`` as::

    long factoryId = action.getLong("factory_entity_id", -1L);
    String unitTypeName = action.getString("unit_type", "tank");
    actionOutcomes.add(produceUnit(factoryId, unitTypeName) ? null : "PRODUCE_UNAVAILABLE");

so the index indirection was never necessary. The curriculum knows exactly which
factory and which unit it wants; it should say so.

What this patch does
--------------------
* adds ``_send_direct_order``: dispatch an already-decoded order dict, reusing the
  same accounting (``curriculum_order``) as ``_send_curriculum_order``;
* rewrites the production stage to pick, from one factory's own menu, the highest
  value-per-credit unit that is **affordable right now** (price read from the
  observation), and emit ``{"kind": "produce_unit", "factory_entity_id": ...,
  "unit_type": ...}``.

Usage::

    python scripts/patch_direct_produce.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "src" / "ironzero" / "rwpp" / "env.py"

OLD = '''        # Pick the target index from the factory's OWN menu, not from a constant.
        #
        # This used to pin the target head at ``target_slots - 1``, and
        # ``decode_action`` resolves that index modulo ``produce_options(state)`` --
        # the (factory, unit) cross product over **every** factory, including the
        # command centre. Measured on Lake at difficulty 1
        # (``.runtime/queue_items.json``): factory 29 was ordered to build
        # ``scout``, a unit that appears in the *command centre's* menu and not in
        # the land factory's, so the engine queued it and never started it --
        # ``queued[].progress`` stayed at exactly **-1.0** ("not started") from tick
        # 1001 to 2751 while ``can_produce`` listed only
        # ``[builder, hoverTank, tank]``.
        #
        # A queue that never drains is worse than an empty one: it holds the
        # ``production_queue_depth`` slots, so this stage reads ``queue_full`` on
        # every subsequent decision and never orders the units that *would* work.
        # The measured consequence was ``factories=4``, ``queue=3``, ``funds=1624``
        # and ``army=0`` for an entire episode.
        #
        # ``produce_options`` already filters ``builder`` out, so any entry it
        # returns is a combat unit this factory can actually make. Selecting the
        # index of the *first option belonging to a combat factory* keeps the
        # stage's "one order per decision" behaviour while guaranteeing the order is
        # executable.
        options = produce_options(state)
        combat_ids = {
            int(entry.get("entity_id", -1))
            for entry in combat_factories
            if entry.get("entity_id") is not None
        }
        # Prefer the best *value per credit* a real factory can make, not the first
        # entry in the menu and not the single strongest unit.
        #
        # ``produce_options`` walks factories in payload order and, inside each, the
        # menu order, so "first entry" is really "whatever the command centre lists
        # first" -- measured as ``(23, scout)`` while three land factories sat idle.
        # Ranking by the unit's combat value is what turns three factories into
        # tanks instead of six scouts.
        target_index = None
        best_score = None
        for index, (factory_id, unit_name) in enumerate(options):
            if factory_id not in combat_ids:
                continue
            # Value per credit: a tank that gets built beats an artillery that the
            # bank can never afford. See ``_unit_value_per_credit``.
            score = _unit_value_per_credit(str(unit_name))
            if best_score is None or score > best_score:
                best_score = score
                target_index = index
        if target_index is None:
            # No producible combat unit on any factory this decision. Returning
            # here is correct: ordering anything else would queue a unit the
            # factory cannot build and wedge the queue (see above).
            return []
        action = np.array(
            [ACTION_KINDS.index("produce_unit"), int(self.micro.groups),
             int(self.micro.position_grid) // 2, int(self.micro.position_grid) // 2,
             int(target_index)],
            dtype=np.int64,
        )
        orders = self._send_curriculum_order(action, {"kind": "produce_unit"})
        if not orders:
            return []
        self.production_curriculum_done = False
        return orders'''

NEW = '''        # Name the factory and the unit explicitly; never route through an index.
        #
        # The index form was the source of two separate wedges. First the target head
        # was pinned at ``target_slots - 1``, which resolved modulo a list covering
        # every factory including the command centre, so a land factory was ordered
        # to build ``scout`` -- absent from its menu -- and the queue item sat at
        # progress -1.0 from tick 1001 to 2751. Then the index was chosen from
        # ``produce_options``, which is rebuilt from the live observation and *shrinks*
        # whenever a factory is busy: measured with scripts/probe_produce_options.py,
        # 13 entries at step 83 and 6 at step 85, so a fixed index names a different
        # pair one decision later and the queue ended up holding ``c_tank``.
        #
        # The bridge accepts the concrete form directly
        # (``action.getLong("factory_entity_id")`` + ``action.getString("unit_type")``),
        # so there is no reason to express a decision the curriculum already knows.
        funds = float((state.get("global_state") or {}).get("funds") or 0.0)
        chosen_factory = None
        chosen_unit = None
        best_score = None
        for entry in combat_factories:
            factory_id = entry.get("entity_id")
            if factory_id is None:
                continue
            menu = [str(name) for name in (entry.get("can_produce") or [])]
            prices = entry.get("can_produce_price") or []
            for position, name in enumerate(menu):
                price = None
                if position < len(prices):
                    try:
                        price = float(prices[position])
                    except (TypeError, ValueError):
                        price = None
                if price is None:
                    price = float(_UNIT_PRICE.get(name.split(".")[-1], 500))
                # Affordability is part of the decision: a 900-credit artillery that
                # the bank cannot pay for is worth less than a 350-credit tank that
                # gets built. Ordering the unaffordable one measured as an *empty*
                # army over a full episode.
                if price > funds:
                    continue
                score = _unit_value_per_credit(name)
                if score <= 0.0:
                    continue
                if best_score is None or score > best_score:
                    best_score = score
                    chosen_factory = int(factory_id)
                    chosen_unit = name
        if chosen_factory is None or chosen_unit is None:
            # Nothing buildable and affordable this decision. Returning here is
            # correct: ordering anything else wedges the queue (see above).
            return []
        orders = self._send_direct_order(
            {
                "kind": "produce_unit",
                "factory_entity_id": int(chosen_factory),
                "unit_type": str(chosen_unit),
            },
            {"kind": "produce_unit", "factory_entity_id": int(chosen_factory),
             "unit_type": str(chosen_unit)},
        )
        if not orders:
            return []
        self.production_curriculum_done = False
        return orders'''

HELPER_ANCHOR = '''    def _send_curriculum_order(
        self, action: np.ndarray, payload: dict[str, Any]
    ) -> list[dict[str, Any]]:'''

HELPER_NEW = '''    def _send_direct_order(
        self, order: dict[str, Any], payload: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Dispatch an already-decoded order, bypassing the action-vector round trip.

        Some curriculum decisions name a concrete entity ("queue a tank at factory
        27"). Encoding that as a head index and decoding it again is not just
        redundant: the index space is rebuilt from the live observation and changes
        size whenever a factory is busy, so the same index names a different pair one
        decision later. Measured consequence: a land factory was ordered to build a
        unit absent from its menu, the queue item never started (progress -1.0), and
        the stage read ``queue_full`` for 771 of 900 decisions with the army at zero.

        Returns a single-element list so callers keep the same "orders to append"
        contract as :meth:`_send_curriculum_order`.
        """

        if not isinstance(order, dict) or not order.get("kind"):
            return []
        self.curriculum_order = {**payload, "accepted": True}
        return [order]

    def _send_curriculum_order(
        self, action: np.ndarray, payload: dict[str, Any]
    ) -> list[dict[str, Any]]:'''


def main() -> int:
    src = PATH.read_text(encoding="utf-8")

    if "_send_direct_order" in src:
        print("already patched")
        return 0

    assert src.count(OLD) == 1, f"production block occurrences: {src.count(OLD)}"
    src = src.replace(OLD, NEW)

    assert src.count(HELPER_ANCHOR) == 1, f"helper anchor occurrences: {src.count(HELPER_ANCHOR)}"
    src = src.replace(HELPER_ANCHOR, HELPER_NEW)

    PATH.write_text(src, encoding="utf-8", newline="\n")
    print("patched production to emit concrete (factory, unit) orders")
    return 0


if __name__ == "__main__":
    sys.exit(main())
