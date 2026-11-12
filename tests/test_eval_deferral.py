# SPDX-License-Identifier: Apache-2.0
"""Evaluation must not be silently skipped when it comes due.

Regression guard for a real training failure: a 4-episode evaluation took
~112 s while the 10 training updates it gated took ~15 s. Every time the
background worker was still busy the trainer printed "skipped" and moved on, so
3 of every 4 due evaluations never ran. A 400-update run reported 4 evaluation
points for 20 due -- the curve was mostly interpolation.

The invariant: a due evaluation is *deferred*, never dropped, and the deferred
update is what eventually gets evaluated. These tests drive
``PPOTrainer._claim_eval_slot`` directly -- the real decision point -- so the
assertions fail if the production logic regresses.
"""

from __future__ import annotations

import threading

import pytest

from ironzero.training.ppo import PPOTrainer


class _Alive:
    """Stands in for a still-running background evaluation worker."""

    def is_alive(self) -> bool:
        return True


class _Dead:
    def is_alive(self) -> bool:
        return False


def _bare_trainer(worker=None) -> PPOTrainer:
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer._eval_lock = threading.Lock()
    trainer._eval_thread = worker
    trainer._in_flight_eval_update = 10 if worker is not None else None
    trainer._pending_eval_update = None
    return trainer


# ----------------------------------------------------------------------
# Free worker: evaluate immediately
# ----------------------------------------------------------------------
def test_free_worker_evaluates_the_current_update() -> None:
    trainer = _bare_trainer(_Dead())
    assert trainer._claim_eval_slot(20) == 20


def test_no_worker_ever_started_evaluates_immediately() -> None:
    trainer = _bare_trainer(None)
    assert trainer._claim_eval_slot(40) == 40


# ----------------------------------------------------------------------
# Busy worker: defer, never drop
# ----------------------------------------------------------------------
def test_busy_worker_defers_instead_of_skipping() -> None:
    trainer = _bare_trainer(_Alive())
    claimed = trainer._claim_eval_slot(20)
    assert claimed is None, "a due evaluation was dropped instead of deferred"
    assert trainer._pending_eval_update == 20


def test_repeated_deferrals_coalesce_to_the_newest_update() -> None:
    """Only the newest pending state is worth evaluating when the worker frees."""

    trainer = _bare_trainer(_Alive())
    for update in (20, 40, 60):
        assert trainer._claim_eval_slot(update) is None
    assert trainer._pending_eval_update == 60


def test_older_update_never_overwrites_a_newer_pending_one() -> None:
    trainer = _bare_trainer(_Alive())
    trainer._pending_eval_update = 100
    assert trainer._claim_eval_slot(40) is None
    assert trainer._pending_eval_update == 100


# ----------------------------------------------------------------------
# Draining the deferral
# ----------------------------------------------------------------------
def test_freed_worker_evaluates_the_deferred_update_not_the_current_one() -> None:
    """The deferred weights are older, but they are what was promised."""

    trainer = _bare_trainer(_Alive())
    trainer._claim_eval_slot(20)          # deferred
    assert trainer._pending_eval_update == 20

    trainer._eval_thread = _Dead()        # worker finished
    assert trainer._claim_eval_slot(40) == 20, (
        "the deferred update was not the one evaluated next"
    )
    assert trainer._pending_eval_update is None, "the deferral was not cleared"


def test_after_draining_the_next_call_is_immediate_again() -> None:
    trainer = _bare_trainer(_Dead())
    trainer._pending_eval_update = 20
    assert trainer._claim_eval_slot(40) == 20
    assert trainer._claim_eval_slot(60) == 60


@pytest.mark.parametrize("deferred", [20, 40, 80, 120])
def test_every_deferred_update_is_eventually_returned(deferred: int) -> None:
    trainer = _bare_trainer(_Alive())
    trainer._claim_eval_slot(deferred)
    trainer._eval_thread = _Dead()
    assert trainer._claim_eval_slot(deferred + 20) == deferred


# ----------------------------------------------------------------------
# The end-of-run drain
# ----------------------------------------------------------------------
def test_pending_eval_starts_unset() -> None:
    assert _bare_trainer(None)._pending_eval_update is None


def test_deferral_survives_until_the_run_drains_it() -> None:
    """Training-end reads ``_pending_eval_update`` to run the last missed eval."""

    trainer = _bare_trainer(_Alive())
    trainer._claim_eval_slot(60)
    assert trainer._pending_eval_update == 60
    # The drain path consumes and clears it exactly once.
    deferred = trainer._pending_eval_update
    trainer._pending_eval_update = None
    assert deferred == 60
    assert trainer._pending_eval_update is None
