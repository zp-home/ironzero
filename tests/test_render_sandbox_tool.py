# SPDX-License-Identifier: Apache-2.0
"""The documentation GIF renderer must stay reproducible and asset-faithful.

``tools/render_sandbox.py`` produced the sandbox clip in the README and on the
Pages site. It is a documentation tool rather than part of the platform, so these
tests pin the properties that would otherwise rot silently:

* the camera picks the fight (the reason the first attempts produced empty lake
  shots) instead of the bounding box or a stack of buildings;
* tile and sprite resolution follow the same lookup order as the browser code, so
  the picture keeps matching the dashboard;
* the committed artefacts still exist at the paths both documents reference, and
  stay small enough to be worth shipping.
"""

from __future__ import annotations

import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
TOOL = REPO / "tools" / "render_sandbox.py"
GIF = REPO / "docs" / "assets" / "sandbox-preview.gif"
STILL = REPO / "docs" / "assets" / "sandbox-still.png"
README = REPO / "README.md"
SITE = REPO / "site"


def _source() -> str:
    return TOOL.read_text(encoding="utf-8")


def test_renderer_exists_and_is_documented() -> None:
    assert TOOL.is_file(), "the renderer referenced by the README is missing"
    source = _source()
    # The docstring has to explain why this exists instead of a browser screenshot.
    assert "mojo" in source.lower() or "named pipes" in source.lower(), (
        "the module no longer explains why a Python renderer is used"
    )
    assert "SPDX-License-Identifier" in source.splitlines()[0]


def test_sprite_lookup_matches_the_browser_order() -> None:
    """``sprites.js`` resolves byType -> aliases -> builder/building/mobile.

    If this drifts, the GIF shows different units than the dashboard for the same
    replay, which is exactly the kind of mismatch the tool exists to avoid.
    """

    source = _source()
    assert "by_type" in source and "aliases" in source and "fallback" in source
    # The order inside entry_for matters: exact type first, then the alias.
    body = re.search(r"def entry_for\(.*?\n    def ", source, re.S)
    assert body, "entry_for not found"
    text = body.group(0)
    assert text.index("unit_type in self.by_type") < text.index("self.aliases.get(unit_type)")
    assert text.index("self.aliases.get(unit_type)") < text.index('self.fallback.get("builder")')


def test_ground_layer_is_read_from_layers() -> None:
    """The map detail keeps ``Ground`` inside ``layers``; a top-level read fails."""

    source = _source()
    assert 'detail.get("layers")' in source, "the layer list is not consulted"
    # It must select by layer name, not blindly take index 0 as the ground.
    assert re.search(r'layer\.get\("name".*?\.lower\(\) == "ground"', source), (
        "the ground layer is not identified by name"
    )


def test_camera_anchors_on_damage_not_density() -> None:
    """The lesson from the first attempts: density anchors land on empty water."""

    source = _source()
    assert "def camera_for_clip" in source
    assert "damage" in source
    # Recomputing per frame makes the ground swim and inflates the GIF; the clip
    # must share one fixed window.
    assert "One fixed window" in source or "fixed window" in source


def test_encoder_prefers_ffmpeg_for_gif_size() -> None:
    """Per-frame Pillow quantisation produced a 7 MB GIF for the same clip."""

    source = _source()
    assert "palettegen" in source and "paletteuse" in source
    assert "sierra2_4a" in source
    # A Pillow fallback must remain, so the tool works without ffmpeg.
    assert "quantize" in source


@pytest.mark.skipif(not GIF.is_file(), reason="GIF asset not present")
def test_documentation_gif_exists_and_is_lean() -> None:
    size_kb = GIF.stat().st_size / 1024
    assert 100 <= size_kb <= 3072, (
        f"{size_kb:.0f} KB is outside the 100 KiB - 3 MiB budget for a README asset"
    )
    # GIF89a magic + a logical screen descriptor.
    header = GIF.read_bytes()[:6]
    assert header in (b"GIF89a", b"GIF87a"), f"not a GIF: {header!r}"


@pytest.mark.skipif(not GIF.is_file(), reason="GIF asset not present")
def test_gif_actually_animates() -> None:
    """A single-frame 'GIF' would silently document nothing."""

    pillow = pytest.importorskip("PIL.Image", reason="Pillow is needed to inspect the GIF")
    with pillow.open(GIF) as image:
        frames = getattr(image, "n_frames", 1)
        width, height = image.size
    assert frames > 20, f"only {frames} frames; the clip is not really animated"
    assert 300 <= width <= 1200, f"unexpected width {width}"
    assert 200 <= height <= 800, f"unexpected height {height}"


@pytest.mark.skipif(not STILL.is_file(), reason="still asset not present")
def test_still_frame_is_present() -> None:
    assert STILL.stat().st_size > 10_000, "the still image looks empty"


def test_readme_embeds_the_clip() -> None:
    text = README.read_text(encoding="utf-8")
    assert "docs/assets/sandbox-preview.gif" in text, "README no longer shows the clip"
    # The caption must be honest about provenance: rendered, not a screenshot.
    assert "render_sandbox.py" in text, "README must name the tool that produced it"


def test_site_publishes_the_clip() -> None:
    """Pages only uploads ``site/``, so the asset has to live inside it."""

    page = (SITE / "index.html").read_text(encoding="utf-8")
    assert "assets/sandbox-preview.gif" in page, "the site lost the sandbox clip"
    assert (SITE / "assets" / "sandbox-preview.gif").is_file(), (
        "the GIF is referenced but not checked in under site/assets"
    )
    assert 'id="sandbox"' in page, "the sandbox section anchor disappeared"


def test_site_section_numbers_stay_sequential() -> None:
    """Inserting a section without renumbering reads as a mistake."""

    page = (SITE / "index.html").read_text(encoding="utf-8")
    numbers = [int(m.group(1)) for m in re.finditer(
        r'class="section-index">(\d{2}) / ', page)]
    assert numbers, "no numbered sections found"
    assert numbers == list(range(1, len(numbers) + 1)), (
        f"section indexes are {numbers}; expected 1..{len(numbers)}"
    )


def test_site_css_defines_the_sandbox_classes() -> None:
    page = (SITE / "index.html").read_text(encoding="utf-8")
    css = (SITE / "styles.css").read_text(encoding="utf-8")
    section = re.search(
        r'<section class="section section-shell sandbox-section".*?</section>', page, re.S
    )
    assert section, "sandbox section markup not found"
    used = set()
    for match in re.finditer(r'class="([^"]+)"', section.group(0)):
        used.update(match.group(1).split())
    defined = set(re.findall(r"\.([a-zA-Z][\w-]*)", css))
    missing = sorted(name for name in used if name not in defined)
    assert not missing, f"classes with no CSS rule: {missing}"
