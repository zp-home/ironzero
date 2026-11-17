# SPDX-License-Identifier: Apache-2.0
"""The PPO form, its JS fallbacks and its state defaults must agree.

All three carry the same "current value" for every control:

* ``index.html``  — the `value=` / `selected` option the user sees;
* ``ppoRun.js``   — the ``clampInt(..., fallback)`` used when the DOM value is
  missing or unparseable;
* ``ppo.js``      — the initial ``state.fe.ppo`` fields the KPI/summary lines read.

They had drifted apart in every direction (updates 40/20, rollout 64/20, ticks
4800/3600, frame_ms 1/2, vision 0/260, eval_every 10/0, eval_episodes 4/1), so a
run could display one configuration and execute another. These tests pin the
three sources to each other and to the trainer's actual units.
"""

from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
HTML_PATH = REPO / "web" / "frontend" / "index.html"
PPORUN_PATH = REPO / "web" / "frontend" / "js" / "ppoRun.js"
PPO_PATH = REPO / "web" / "frontend" / "js" / "ppo.js"
FEINFER_RUN_PATH = REPO / "web" / "frontend" / "js" / "feInferRun.js"
STATE_PATH = REPO / "web" / "frontend" / "js" / "state.js"

HTML = HTML_PATH.read_text(encoding="utf-8")
PPORUN = PPORUN_PATH.read_text(encoding="utf-8")
PPO = PPO_PATH.read_text(encoding="utf-8")
FEINFER_RUN = FEINFER_RUN_PATH.read_text(encoding="utf-8")
STATE = STATE_PATH.read_text(encoding="utf-8")

#: html id -> camelCase key used by ``els``
CONTROLS = {
    "fe-ppo-updates": "fePpoUpdates",
    "fe-ppo-envs": "fePpoEnvs",
    "fe-ppo-rollout": "fePpoRollout",
    "fe-ppo-ticks": "fePpoTicks",
    "fe-ppo-frame-ms": "fePpoFrameMs",
    "fe-ppo-fast-forward": "fePpoFastForward",
    "fe-ppo-vision": "fePpoVision",
    "fe-ppo-eval-every": "fePpoEvalEvery",
    "fe-ppo-eval-episodes": "fePpoEvalEpisodes",
}

#: html id -> camelCase key, for the live-inference page (``feInferRun.js``).
INFER_CONTROLS = {
    "fe-frame-ms": "feFrameMs",
    "fe-fast-forward": "feFastForward",
}


def html_default(html_id: str) -> str:
    """The value the user sees: an input's ``value=`` or a select's selected option."""

    tag = re.search(rf'<(input|select)[^>]*id="{html_id}"[^>]*>', HTML)
    assert tag, f"{html_id} not found in index.html"
    if tag.group(1) == "input":
        attrs = tag.group(0)
        value = re.search(r'value="([^"]*)"', attrs)
        assert value, f"{html_id} has no value="
        return value.group(1)
    start = HTML.index(tag.group(0)) + len(tag.group(0))
    body = HTML[start:HTML.index("</select>", start)]
    selected = re.search(r'<option value="([^"]*)"[^>]*\bselected\b', body)
    assert selected, f"{html_id} has no option marked selected"
    return selected.group(1)


def js_fallback(key: str) -> int:
    """The 4th argument of ``clampInt(els.<key>...)``."""

    match = re.search(rf"clampInt\(els\.{key}[^;]*?,\s*(-?\d+)\s*\)", PPORUN)
    assert match, f"no clampInt fallback found for {key}"
    return int(match.group(1))


def infer_js_fallback(key: str) -> int:
    """``feInferRun.js`` reads its pacing controls through the same helper."""

    match = re.search(rf"clampInt\(els\.{key}[^;]*?,\s*(-?\d+)\s*\)", FEINFER_RUN)
    assert match, f"no clampInt fallback found for {key} in feInferRun.js"
    return int(match.group(1))


@pytest.mark.parametrize("html_id,key", sorted(CONTROLS.items()))
def test_js_fallback_equals_the_form_default(html_id: str, key: str) -> None:
    """A missing DOM value must not silently switch to a different config."""

    assert js_fallback(key) == int(html_default(html_id)), (
        f"{html_id}: form shows {html_default(html_id)} but the JS fallback is "
        f"{js_fallback(key)}; a run would execute a different config than displayed"
    )


def test_state_fog_default_matches_the_form() -> None:
    """``state.fe.ppo.visionRadius`` drives the KPI text; it must match the form."""

    match = re.search(r"ppo:\s*\{(.*?)\n    \}", PPO, re.S)
    assert match, "initFePpoState block not found"
    block = match.group(1)
    vision = int(re.search(r"visionRadius:\s*(\d+)", block).group(1))
    frame = int(re.search(r"frameMs:\s*(\d+)", block).group(1))
    assert vision == int(html_default("fe-ppo-vision"))
    assert frame == int(html_default("fe-ppo-frame-ms"))


def test_state_object_has_no_duplicate_keys() -> None:
    """A repeated key silently wins; that is how vision 260 outlived the form's 0."""

    match = re.search(r"ppo:\s*\{(.*?)\n    \}", PPO, re.S)
    assert match
    keys = re.findall(r"^\s+(\w+):", match.group(1), re.M)
    duplicates = {key for key in keys if keys.count(key) > 1}
    assert not duplicates, f"duplicate keys in state.fe.ppo: {sorted(duplicates)}"


def test_default_episode_is_long_armies_meet() -> None:
    """The default tick count must outlast the armies' approach on every arena.

    Measured spawn separation (``configs/maps/IronZero*.meta.json``) divided by
    the calibrated tank speed gives the ticks needed just to make contact; a
    default shorter than that can only ever produce ``timeout`` and a flat
    reward, which is exactly the "训练完全不收敛" symptom.

    The form default is now ``0``, which means "derive the cap from the map"
    (a fixed constant cannot be right for both a 1600 px duel and a 25v25 arena,
    and the old 4800 cut every original-map episode off before the engine could
    decide anything). ``TRAVERSALS_PER_EPISODE`` budgets several map crossings,
    so the derived value is held to the same contact-time floor here.
    """

    tick_rate = 60
    tank_px_per_s = 66.0  # micro_config.unit_speed
    default_ticks = int(html_default("fe-ppo-ticks"))
    assert default_ticks == 0, "0 is the documented 'derive from the map' sentinel"

    from ironzero.rwpp.config import recommended_episode_ticks

    for units in (5, 10, 15, 20, 25):
        meta_path = REPO / "configs" / "maps" / f"IronZero{units}.meta.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        separation = meta["spawn"]["spacing"]["min_between_teams_px"]
        width = float(meta.get("map", {}).get("width_px") or 2700.0)
        height = float(meta.get("map", {}).get("height_px") or 2500.0)
        contact_ticks = separation / tank_px_per_s * tick_rate
        derived = recommended_episode_ticks(width, height, 2)
        assert contact_ticks < derived, (
            f"IronZero{units}: armies need {contact_ticks:.0f} ticks to meet but the "
            f"map-derived episode is only {derived}; every episode would time out"
        )


def test_form_does_not_claim_fog_the_default_disables() -> None:
    """The hint text used to say "默认 260px" while the input defaulted to 0."""

    assert int(html_default("fe-ppo-vision")) == 0
    # No copy may advertise a non-zero default fog radius any more.
    assert "默认开启 260px" not in HTML
    assert "观测迷雾 260px" not in HTML


def test_warmstart_checkpoints_were_trained_without_fog() -> None:
    """Defaulting fog on would hand a warm start an observation it never saw."""

    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(REPO / "src"))
    for name in ("bc_ironzero10", "bc_ironzero25"):
        path = REPO / "runs" / "experiments" / name / "checkpoint.pt"
        if not path.is_file():
            pytest.skip(f"{name} weights not present")
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        vision = (payload.get("micro_config") or {}).get("vision_radius")
        assert vision in (0, 0.0, None), (
            f"{name} was trained with vision_radius={vision}; the form default must "
            f"match it or warmstarting starts from a mismatched observation"
        )


@pytest.mark.parametrize("html_id,key", sorted(INFER_CONTROLS.items()))
def test_infer_js_fallback_equals_the_form_default(html_id: str, key: str) -> None:
    """The live-inference page must read its own controls, not a hard-coded value.

    ``feInferRun.js`` used to do ``els.feFrameMs ? ... : 2`` while ``fe-frame-ms``
    did not exist in ``index.html``, so **every** live-inference run was silently
    pinned to 2 ms = 6.5x. The user could not see or change it. Now the select is
    real, ``els`` caches it, and the fallback must equal what the form shows.
    """

    assert infer_js_fallback(key) == int(html_default(html_id)), (
        f"{html_id}: form shows {html_default(html_id)} but feInferRun.js falls "
        f"back to {infer_js_fallback(key)}; the run would use a different pacing"
    )


@pytest.mark.parametrize("html_id,key", sorted(INFER_CONTROLS.items()))
def test_infer_controls_are_cached_and_sent(html_id: str, key: str) -> None:
    """A control that is not in ``els`` is invisible to the code that reads it.

    This is the exact shape of the original bug: ``feInferRun.js`` asked for
    ``els.feFrameMs``, ``state.js`` never cached ``fe-frame-ms``, and the ``?:``
    fallback hid it. All three links (HTML id, ``els`` cache, request body) are
    pinned here.
    """

    assert f'id="{html_id}"' in HTML
    assert f"'{html_id}'" in STATE, f"{html_id} must be cached in els (state.js)"


def test_infer_request_carries_pacing_and_fast_forward() -> None:
    """Both values must reach ``/infer``, or the controls are decorative."""

    assert "frame_ms: frameMs" in FEINFER_RUN
    assert "fast_forward: fastForward" in FEINFER_RUN
    # No hard-coded pacing may survive anywhere in the launch path.
    assert "parseInt(els.feFrameMs.value, 10) : 2" not in FEINFER_RUN
    assert not re.search(r"feFrameMs[^;]*?:\s*2\s*;", FEINFER_RUN)


def test_infer_frame_ms_offers_the_measured_speeds() -> None:
    """The options must be the measured ones, best (unpaced) first."""

    block = HTML.split('id="fe-frame-ms"')[1].split("</select>")[0]
    values = [int(value) for value in re.findall(r'<option value="(\d+)"', block)]
    assert values == [0, 1, 2, 16], values
    assert int(html_default("fe-frame-ms")) == 0, "0 (unpaced, ~26x) is the recommendation"
    # fast_forward stays a separate control: 1 = off is the default.
    ff_block = HTML.split('id="fe-fast-forward"')[1].split("</select>")[0]
    assert [int(value) for value in re.findall(r'<option value="(\d+)"', ff_block)] == [1, 2, 4]


def test_frame_ms_default_matches_the_recommended_option() -> None:
    """The form recommends 0 ms (measured ~26.5x); the JS fallback said 1 ms.

    ``frame_ms=1`` is 10.7x, not the "约 16×" the old label claimed, and 0 is
    free: with ``physics_ms=16`` pinning the delta, the sleep is pure loss.
    """

    assert int(html_default("fe-ppo-frame-ms")) == 0
    block = HTML.split('id="fe-ppo-frame-ms"')[1].split("</select>")[0]
    assert "不限速" in block and "推荐" in block
    # The stale multiplier must be gone from every label and hint.
    assert "约 16×" not in HTML
    assert "约 8×" not in HTML


def test_multipler_copy_matches_the_measured_numbers() -> None:
    """``frame_ms`` copy must quote 0/1/2 ms as ~26x / ~10x / ~6.5x.

    Measured on 6000 real ticks with ``physics_ms=16`` (audit/加速方案_契约.md §0):
    16 ms = 1x, 2 ms = 6.5x, 1 ms = 10.7x, 0 ms = 26.5x.
    """

    for claim in ("约 26×", "约 10×", "约 6.5×"):
        assert claim in HTML, f"index.html never quotes the measured {claim}"


def test_ppo_fast_forward_control_exists_and_is_cached() -> None:
    """``fe-ppo-fast-forward`` must be a real control, cached and sent."""

    assert 'id="fe-ppo-fast-forward"' in HTML
    assert "'fe-ppo-fast-forward'" in STATE
    assert "fast_forward: fastForward" in PPORUN
    assert "fastForward" in PPO, "the summary/KPI line must be able to show it"


class StaleCopyTests(unittest.TestCase):
    """Panel copy that contradicted the form it sits next to."""

    def test_mode_note_is_not_hard_coded_to_one_map(self) -> None:
        """The banner said "Artillery 挑战图（22 vs 25 AI）" whatever map was picked.

        It now goes through ``fePpoScenarioLabel()``, so the header and the map
        select directly below it can no longer disagree.
        """

        # Only code lines matter; the explanatory comment above the fix quotes the
        # old string on purpose.
        code_lines = [
            line for line in PPO.splitlines()
            if "Artillery 挑战图（22 vs 25 AI）" in line
            and not line.lstrip().startswith(("//", "*", "/*"))
        ]
        assert not code_lines, f"hard-coded scenario copy still in code: {code_lines}"
        assert "fePpoScenarioLabel" in PPO
        # And the label helper reads the live form rather than a constant.
        block = PPO.split("export function fePpoScenarioLabel")[1].split("\n}")[0]
        assert "els.fePpoMap" in block

    def test_log_status_reads_the_form_before_a_job_exists(self) -> None:
        """It used to show ``eval=最后/1`` while the form said 每 10 轮 / 4 局."""

        block = PPO.split("export function feRenderPpoLog")[1].split("\n}")[0]
        # The pre-job branch must consult the form, not the stale state fields.
        assert "formInt(" in block, (
            "the idle log line must read the live form; state.fe.ppo only holds "
            "defaults until a job starts"
        )
        for key in ("fePpoEvalEvery", "fePpoEvalEpisodes", "fePpoFrameMs", "fePpoVision"):
            assert key in block, f"{key} missing from the log status line"

    def test_form_int_helper_is_defined(self) -> None:
        """An undefined helper here would throw on every dashboard render."""

        assert "function formInt(" in PPO
        assert "numOrNull" in PPO.split("function formInt(")[1].split("\n}")[0]


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
