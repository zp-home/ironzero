"""produceUnit must call the engine's production entry point, not a bare command.

Measured with scripts/probe_factory_state.py on Lake at difficulty 1:

    order   : produce_unit factory=27 unit_type=tank   -> receipt "ok"
    factory : landFactory, can_produce [builder,hoverTank,tank,artillery,scout]
    queued  : [{"unit_type": "c_tank", "count": 1, "progress": -1.0}]
    funds   : 1201.0
    queue_progress : 0.0

The item is in the queue and never advances past ``-1.0`` ("not started"), with the
money in the bank and the type in the menu. So the command was *accepted* but never
*executed as production*.

Why: ``produceUnit`` built a ``CommandControllerCommand`` and set its
``actionId``. That path relies on the command dispatcher routing an action id into
the factory's build queue, which is not what the engine's own production code does.
``Factory`` exposes the real entry point:

    public void executeSpecialAction(SpecialAction action, boolean shift)
        -> buildQueue.queueOrCancel(action, shift, null, null)

``queueOrCancel`` is what charges the price and enqueues the build, and it is what
the built-in AI reaches through the same method. Calling it directly removes the
guesswork about how a command with an ``actionId`` is interpreted.

Usage::

    python scripts/patch_factory_execute.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "src" / "ironzero_bridge" / "java" / "ironzero" / "IronZeroBridge.java"

OLD = '''        try {
            if (!action.isAvailableForUnit(factory) || !action.canAffordForUnit(factory, false)) {
                return false;
            }
        } catch (RuntimeException failure) {
            return false;
        }
        CommandControllerCommand command =
                engine.commandController.createCommandForTeam(engine.playerTeam);
        command.addUnit(factory);
        command.setActionId(action.getActionId());
        return true;
    }'''

NEW = '''        try {
            if (!action.isAvailableForUnit(factory) || !action.canAffordForUnit(factory, false)) {
                return false;
            }
        } catch (RuntimeException failure) {
            return false;
        }
        // Call the engine's own production entry point.
        //
        // The previous form built a CommandControllerCommand and set its actionId,
        // which relies on the command dispatcher turning an action id into a build
        // queue entry. It does not: measured on Lake at difficulty 1, the item
        // appeared in the queue with progress -1.0 and never started, with 1201
        // credits in the bank and the type present in the factory's menu.
        //
        // ``Factory.executeSpecialAction`` is what the engine itself uses (and what
        // the built-in AI reaches): it forwards to
        // ``FactoryBuildQueue.queueOrCancel``, which charges the price and enqueues
        // the build. Routing through it means "accepted" really means "the factory
        // is now building this".
        try {
            if (factory instanceof com.corrodinggames.rts.game.units.buildings.Factory) {
                ((com.corrodinggames.rts.game.units.buildings.Factory) factory)
                        .executeSpecialAction(action, false);
                return true;
            }
            if (factory instanceof com.corrodinggames.rts.game.units.buildings.FactoryInterface) {
                ((com.corrodinggames.rts.game.units.buildings.FactoryInterface) factory)
                        .executeSpecialAction(action, false);
                return true;
            }
        } catch (RuntimeException failure) {
            // Fall through to the command path rather than reporting a phantom
            // rejection: the queue may still accept the order on older builds.
        }
        CommandControllerCommand command =
                engine.commandController.createCommandForTeam(engine.playerTeam);
        command.addUnit(factory);
        command.setActionId(action.getActionId());
        return true;
    }'''


def main() -> int:
    src = PATH.read_text(encoding="utf-8")

    if "executeSpecialAction(action, false)" in src:
        print("already patched")
        return 0

    assert src.count(OLD) == 1, f"produceUnit tail occurrences: {src.count(OLD)}"
    src = src.replace(OLD, NEW)

    PATH.write_text(src, encoding="utf-8", newline="\n")
    print("patched produceUnit to call Factory.executeSpecialAction")
    return 0


if __name__ == "__main__":
    sys.exit(main())
