# SPDX-License-Identifier: Apache-2.0
"""The sandbox render/ingest optimisations must not change what is drawn.

Reported symptom: sandbox playback stuttered during inference. The back-end fix
(``jsonl_index``) removed a per-poll whole-file parse; these cover the *frontend*
half, which lives in browser ES modules and therefore cannot be imported by
pytest. Instead of copying the logic into Python -- which would only test the
copy -- the real modules are executed under Node and compared against the
previous implementation, kept verbatim in ``tests/js/frontend_checks.mjs``.

What is pinned:

* ``thinPoints`` bounds a series without moving either endpoint (the chart
  downsampler that stops 30 000-point redraws);
* ``interpolateUnits`` produces byte-identical fields to the old per-frame
  allocation, while now reusing pooled objects;
* ``isUnitVisible`` and ``updateUnitAngles`` agree with the old ``hypot``-based
  O(n^2) scans on a dense grid / randomised frame, after the squared-distance and
  per-team-bucket rewrites;
* every frontend module still parses, since the page loads them directly with no
  build step to catch a typo.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
JS_DIR = REPO / "tests" / "js"
FRONTEND_JS = REPO / "web" / "frontend" / "js"

NODE = shutil.which("node")


def _node_available() -> bool:
    return NODE is not None


requires_node = pytest.mark.skipif(
    not _node_available(),
    reason="node is needed to execute the browser ES modules under test",
)


def _run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [NODE, "--experimental-vm-modules", str(JS_DIR / script), *args],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

@requires_node
def test_frontend_behaviour_checks_pass() -> None:
    """The behavioural equivalence checks all pass (see the .mjs for the list)."""

    result = _run("frontend_checks.mjs")
    assert result.returncode == 0, (
        "frontend checks failed\n"
        f"stdout:\n{result.stdout[-4000:]}\n"
        f"stderr:\n{result.stderr[-2000:]}"
    )
    payload = json.loads(result.stdout)
    assert payload["failed"] == 0, json.dumps(payload, indent=2, ensure_ascii=False)
    # Guard against the whole suite silently becoming a no-op.
    assert len(payload["results"]) >= 10, (
        f"only {len(payload['results'])} checks ran; the file probably stopped early"
    )
    names = {item["name"] for item in payload["results"]}
    for required in (
        "thinPoints bounds a large series and preserves both ends",
        "interpolateUnits matches the previous implementation",
        "isUnitVisible agrees with the previous hypot implementation",
        "updateUnitAngles matches the previous nearest-enemy scan",
        "updateUnitAngles no longer calls Math.hypot in the nearest-enemy scan",
    ):
        assert required in names, f"missing check: {required}"

    # The nearest-enemy rewrite must be shown to *act*, not merely to agree:
    # with one team there is no enemy to find and every assertion passes vacuously.
    aimed = next(
        item for item in payload["results"]
        if item["name"] == "updateUnitAngles matches the previous nearest-enemy scan"
    )
    assert aimed["detail"]["aimed"] == aimed["detail"]["units"], (
        f"only {aimed['detail']['aimed']}/{aimed['detail']['units']} units found an enemy"
    )


@requires_node
def test_every_frontend_module_parses() -> None:
    """The page loads these as ES modules directly; nothing else syntax-checks them."""

    result = _run("syntax_check.mjs")
    # Node prints an ExperimentalWarning for --experimental-vm-modules on stderr;
    # only an unparsed module (reported on stdout) is a failure.
    assert "0/" not in result.stdout, result.stdout[-3000:]
    assert "modules parsed" in result.stdout, result.stdout[-3000:]
    assert "FAIL" not in result.stdout, result.stdout[-3000:]
    count = len(list(FRONTEND_JS.glob("*.js")))
    assert f"{count}/{count} modules parsed" in result.stdout, (
        f"expected all {count} modules to parse, got:\n{result.stdout[-2000:]}"
    )


def _code_lines(source: str) -> str:
    """The source with ``//`` comment lines removed.

    Several assertions below are "this risky construct is gone", and the fix's own
    explanatory comments necessarily *name* the construct they replaced. Matching
    raw text would fail on the comments, so strip whole-line comments first.
    """

    kept = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("//") or stripped.startswith("*") or stripped.startswith("/*"):
            continue
        kept.append(line)
    return "\n".join(kept)


def test_frame_draw_calls_stay_at_the_optimised_budget() -> None:
    """Pin the per-frame canvas work on a busy 149-unit frame.

    The optimisations removed a ``shadowBlur`` per unit and an O(n^2) ``hypot``
    scan. Both are easy to reintroduce accidentally (one stray ``ctx.shadowBlur``
    inside the unit loop is enough), and neither is visible in a functional test,
    so the op counts are asserted here.
    """

    if not _node_available():  # pragma: no cover - depends on the host
        pytest.skip("node is needed to count canvas operations")

    proc = subprocess.run(
        [NODE, str(JS_DIR / "count_draw_ops.mjs")],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.returncode == 0, f"{proc.stdout[-3000:]}\n{proc.stderr[-2000:]}"
    payload = json.loads(proc.stdout.split("per frame", 1)[0])
    counts = payload["counts"]

    assert payload["units"] == 149, "the fixture frame changed size"
    # A blurred shadow per unit was the single most expensive op in this layer.
    assert counts["shadowBlurSets"] == 0, (
        f"{counts['shadowBlurSets']} shadowBlur sets per frame; "
        "the per-unit glow is blurring again"
    )
    # The nearest-enemy scan must use squared distances, not hypot.
    assert counts["hypot"] == 0, (
        f"{counts['hypot']} Math.hypot calls per frame; "
        "the O(n^2) distance scan came back"
    )
    # And the replacement must not have bought that by trading fills for blurs:
    # a two-layer glow per unit would roughly double these.
    assert counts["fill"] <= 320, f"{counts['fill']} fills per frame (budget 320)"
    assert counts["stroke"] <= 190, f"{counts['stroke']} strokes per frame (budget 190)"


@requires_node
def test_edge_cases_for_the_bounded_follow_and_chart_throttle() -> None:
    """The caps must not truncate real matches, and the throttle must release."""

    result = _run("edge_cases.mjs")
    assert result.returncode == 0, (
        f"edge-case checks failed\nstdout:\n{result.stdout[-3000:]}\n"
        f"stderr:\n{result.stderr[-1500:]}"
    )
    payload = json.loads(result.stdout)
    assert payload["failed"] == 0, json.dumps(payload, indent=2, ensure_ascii=False)
    names = {item["name"] for item in payload["results"]}
    assert "FE_LOAD_MAX_PAGES covers a longest realistic episode" in names
    assert "chart throttle allows an immediate first draw, then rate-limits" in names


def test_chart_thinning_is_wired_into_the_shared_primitive() -> None:
    """Downsampling must live in ``drawDashLine`` so every chart inherits it."""

    source = (FRONTEND_JS / "control.js").read_text(encoding="utf-8")
    assert "export function thinPoints" in source
    assert "DASH_MAX_POINTS" in source
    # The unconditional sort inside the drawing loop is gone (it ran once per
    # series per redraw on already-ordered data); sorting is now opt-in.
    code = _code_lines(source)
    assert "const points = item.points.slice().sort((left, right) => left.x - right.x);" not in code, (
        "drawDashLine still re-sorts every series on every redraw"
    )
    assert "const points = entry.points;" in code, "the draw loop should use the prepared points"
    assert "cfg.sortPoints" in source, "an opt-in sort path must remain available"


def test_chart_redraws_are_rate_limited() -> None:
    """A 700 ms poll must not force a canvas redraw of a 60 s window every time."""

    source = (FRONTEND_JS / "feInferUi.js").read_text(encoding="utf-8")
    assert "FE_CHARTS_MIN_INTERVAL_MS" in source
    assert "feChartsDue" in source
    assert "if (!feChartsDue(force)) return;" in source


def test_follow_uses_a_bounded_window_but_replay_load_does_not() -> None:
    """Live follow is capped; loading a finished job still fetches everything.

    A single shared setting cannot satisfy both: capping the loader would truncate
    the replay, and leaving the follower unbounded is the original bug.
    """

    fe_infer = (FRONTEND_JS / "feInfer.js").read_text(encoding="utf-8")
    assert "FE_FOLLOW_FRAME_LIMIT" in fe_infer
    # The follow path passes an explicit limit...
    assert "Api.rwppInferFrames(task.jobId, task.fetchCursor || 0, FE_FOLLOW_FRAME_LIMIT)" in fe_infer
    # ...and the loader pages until the cursor stops advancing, so it still ends
    # up with the complete episode.
    assert "FE_LOAD_MAX_PAGES" in fe_infer
    assert "if (task.fetchCursor <= before) break;" in fe_infer

    sandbox = (FRONTEND_JS / "sandbox.js").read_text(encoding="utf-8")
    assert "SANDBOX_FOLLOW_FRAME_LIMIT" in sandbox

    api = (FRONTEND_JS / "api.js").read_text(encoding="utf-8")
    assert "&limit=' + bounded" in api, "the limit must reach the query string"
    assert "if (limit === undefined || limit === null) return cursor;" in api, (
        "omitting limit must keep the whole-tail semantics the loader depends on"
    )


def test_unit_glow_no_longer_uses_a_per_unit_shadow_blur() -> None:
    """``shadowBlur`` is a full blur pass per call; 149 units made it dominate."""

    sprites = _code_lines((FRONTEND_JS / "sprites.js").read_text(encoding="utf-8"))
    hud = _code_lines((FRONTEND_JS / "hud.js").read_text(encoding="utf-8"))
    assert "shadowBlur" not in sprites, "sprites.js still blurs a shadow per unit"
    assert "shadowBlur" not in hud, "hud.js still blurs a shadow per unit/shot"


def test_nearest_enemy_is_bucketed_by_team() -> None:
    """The O(n^2) scan ran every frame for every unit; it must not come back."""

    sprites = (FRONTEND_JS / "sprites.js").read_text(encoding="utf-8")
    assert "function nearestEnemy" in sprites
    assert "byTeam" in sprites
    # The old shape: an inner loop over every unit in the frame.
    assert "for (const item of frame.units) {" not in sprites, (
        "the per-unit full-frame scan is still present"
    )
    # Squared distances instead of a hypot per candidate pair.
    assert "bestSq" in sprites
