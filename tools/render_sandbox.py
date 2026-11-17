# SPDX-License-Identifier: Apache-2.0
"""Render a sandbox-style GIF/PNG from a recorded real-game frame stream.

Why this exists
---------------
The dashboard's sandbox is a browser canvas. Reproducing a *pixel* screenshot of
it needs a browser, and Chrome cannot start under the DSH file sandbox here (its
mojo IPC needs named pipes, e.g.
``platform_channel.cc: Check failed: . : 拒绝访问 (0x5)``). So this renders the
same picture the canvas would draw, from the same recorded data and the same
game assets, in Python:

* the map's real ``Ground`` layer, stitched from the tilesets the backend serves
  at ``/api/assets``;
* real unit sprites from the game's drawable atlases, resolved through the same
  ``byType`` → ``aliases`` → ``fallback`` order as ``sprites.js``;
* the sandbox's conventions: own = blue, enemy = red, HP bar above the unit, a
  camera window, and a HUD line with tick/time and team HP fractions.

It is a *renderer for documentation*, not part of the platform: it reads
``frames.jsonl`` directly and never touches the dashboard.

Usage
-----
    # one still frame
    python tools/render_sandbox.py --stream rwpp-live-rwpp-21203c65 --start 5660 \
        --still docs/media/sandbox-still.png

    # a GIF (needs ffmpeg for the palette pipeline; falls back to Pillow)
    python tools/render_sandbox.py --stream rwpp-live-rwpp-21203c65 --start 5560 \
        --frames 300 --stride 2 --gif docs/media/sandbox-preview.gif

    # find a good clip in a stream
    python tools/render_sandbox.py --stream NAME --survey
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import shutil
import subprocess
import sys
import tempfile
import urllib.request

from PIL import Image, ImageDraw, ImageFont

REPO = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_BACKEND = "http://127.0.0.1:8000"
CACHE_DIR = REPO / "runs" / "_cache" / "render_sandbox"

# Sandbox colours, mirrored from web/frontend/js/{terrain,hud}.js.
OWN_GLOW = (77, 163, 255)
ENEMY_GLOW = (255, 95, 109)
OWN_BODY = (47, 111, 191)
ENEMY_BODY = (181, 48, 60)
BG_TOP = (6, 13, 23)
BG_BOTTOM = (3, 7, 13)

HUD_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)

__all__ = ["main"]


# --------------------------------------------------------------------------- #
# assets
# --------------------------------------------------------------------------- #
def _backend(base: str) -> str:
    return base.rstrip("/")


def fetch_json(base: str, path: str, timeout: float = 60.0) -> dict:
    with urllib.request.urlopen(_backend(base) + path, timeout=timeout) as response:
        return json.loads(response.read().decode())


def fetch_asset(base: str, relative: str) -> pathlib.Path:
    """Download one game asset into the cache, keyed by its repo-relative path."""

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = relative.replace("/", "__").replace("\\", "__")
    target = CACHE_DIR / safe
    if target.is_file() and target.stat().st_size > 0:
        return target
    url = _backend(base) + "/api/assets/" + urllib.request.quote(relative)
    with urllib.request.urlopen(url, timeout=90) as response:
        blob = response.read()
    target.write_bytes(blob)
    return target


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in HUD_FONT_CANDIDATES:
        if pathlib.Path(candidate).is_file():
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
    return ImageFont.load_default()


# --------------------------------------------------------------------------- #
# terrain
# --------------------------------------------------------------------------- #
class Terrain:
    """The map's ground layer, stitched once and reused for every frame."""

    def __init__(self, base: str, detail: dict) -> None:
        self.tile_w = int(detail["tile_width"])
        self.tile_h = int(detail["tile_height"])
        self.cols = int(detail["width"])
        self.rows = int(detail["height"])
        self.map_w = int(detail["width_px"])
        self.map_h = int(detail["height_px"])

        # `Ground` lives inside `layers`; the backend keeps load_map's internals
        # rather than hoisting it to the top level.
        ground = None
        for layer in detail.get("layers") or []:
            if str(layer.get("name", "")).lower() == "ground":
                ground = layer.get("data")
                break
        if ground is None and detail.get("layers"):
            ground = detail["layers"][0].get("data")
        self.ground = ground or []

        # firstgid -> (atlas image, tile columns)
        self.atlases: list[tuple[int, Image.Image, int]] = []
        for tileset in detail.get("tilesets") or []:
            image_rel = tileset.get("image")
            if not image_rel:
                continue
            try:
                image = Image.open(fetch_asset(base, image_rel)).convert("RGBA")
            except Exception:  # noqa: BLE001 - a missing atlas just leaves gaps
                continue
            columns = max(1, image.width // self.tile_w)
            self.atlases.append((int(tileset["firstgid"]), image, columns))
        self.atlases.sort(key=lambda item: item[0])

    def _resolve(self, gid: int) -> tuple[Image.Image, int] | None:
        if gid <= 0:
            return None
        chosen = None
        for firstgid, image, columns in self.atlases:
            if gid >= firstgid:
                chosen = (firstgid, image, columns)
            else:
                break
        if chosen is None:
            return None
        firstgid, image, columns = chosen
        return image, gid - firstgid

    def render(self, crop: dict) -> Image.Image:
        """Draw ``crop`` of the map at 1:1 map pixels."""

        cw = max(1, int(crop["width"]))
        ch = max(1, int(crop["height"]))
        canvas = Image.new("RGBA", (cw, ch), (7, 17, 28, 255))
        cx, cy = int(crop["x"]), int(crop["y"])

        first_col = max(0, cx // self.tile_w)
        first_row = max(0, cy // self.tile_h)
        last_col = min(self.cols - 1, (cx + cw) // self.tile_w)
        last_row = min(self.rows - 1, (cy + ch) // self.tile_h)

        for row in range(first_row, last_row + 1):
            for col in range(first_col, last_col + 1):
                index = row * self.cols + col
                if index >= len(self.ground):
                    continue
                resolved = self._resolve(int(self.ground[index]))
                if resolved is None:
                    continue
                atlas, local = resolved
                tile_cols = max(1, atlas.width // self.tile_w)
                sx = (local % tile_cols) * self.tile_w
                sy = (local // tile_cols) * self.tile_h
                if sx + self.tile_w > atlas.width or sy + self.tile_h > atlas.height:
                    continue
                tile = atlas.crop((sx, sy, sx + self.tile_w, sy + self.tile_h))
                canvas.paste(tile, (col * self.tile_w - cx, row * self.tile_h - cy), tile)
        return canvas


# --------------------------------------------------------------------------- #
# sprites
# --------------------------------------------------------------------------- #
class SpriteBank:
    """Unit sprites resolved exactly like ``sprites.js::resolveUnitSprite``."""

    def __init__(self, base: str, manifest: dict) -> None:
        self.base = base
        self.by_type = manifest.get("byType") or {}
        self.aliases = manifest.get("aliases") or {}
        self.fallback = manifest.get("fallback") or {}
        self._images: dict[str, Image.Image | None] = {}

    def entry_for(self, unit: dict) -> dict | None:
        unit_type = unit.get("unit_type") or ""
        if unit_type and unit_type in self.by_type:
            return self.by_type[unit_type]
        alias = self.aliases.get(unit_type) if unit_type else None
        if alias and alias in self.by_type:
            return self.by_type[alias]
        if unit.get("is_builder") and self.fallback.get("builder"):
            return self.fallback["builder"]
        if unit.get("is_building") and self.fallback.get("building"):
            return self.fallback["building"]
        return self.fallback.get("mobile") or self.fallback.get("building")

    def image_for(self, entry: dict | None) -> Image.Image | None:
        if not entry:
            return None
        rel = entry.get("image")
        if not rel:
            return None
        if rel in self._images:
            return self._images[rel]
        try:
            image = Image.open(fetch_asset(self.base, rel)).convert("RGBA")
            frames = max(1, int(entry.get("frames") or 1))
            frame_w = int(entry.get("frameW") or (image.width // frames))
            frame_h = int(entry.get("frameH") or image.height)
            # Frame 0 only: a documentation clip does not need the animation strip.
            image = image.crop((0, 0, min(frame_w, image.width), min(frame_h, image.height)))
        except Exception:  # noqa: BLE001
            image = None
        self._images[rel] = image
        return image


# --------------------------------------------------------------------------- #
# camera
# --------------------------------------------------------------------------- #
def camera_around(anchor: tuple[float, float], points: list[dict],
                  map_w: int, map_h: int, out_w: int, out_h: int,
                  min_units: int = 32) -> dict:
    """A window centred on ``anchor``, widened until enough units are inside."""

    aspect = out_w / out_h
    half_x = 320.0
    half_y = half_x / aspect
    for _ in range(8):
        inside = sum(
            1 for u in points
            if abs(u["x"] - anchor[0]) <= half_x and abs(u["y"] - anchor[1]) <= half_y
        )
        if inside >= min(min_units, len(points)) or half_x > map_w * 0.42:
            break
        half_x *= 1.25
        half_y = half_x / aspect

    width = max(340.0, min(half_x * 2, map_w))
    height = max(250.0, min(half_y * 2, map_h))
    if width / height > aspect:
        height = min(map_h, width / aspect)
    else:
        width = min(map_w, height * aspect)
    x = max(0.0, min(anchor[0] - width / 2, map_w - width))
    y = max(0.0, min(anchor[1] - height / 2, map_h - height))
    return {"x": x, "y": y, "width": width, "height": height}


def camera_for_clip(frames: list[dict], map_w: int, map_h: int,
                    out_w: int, out_h: int) -> dict:
    """One fixed window for the whole clip, anchored where damage happens.

    Recomputing per frame makes the terrain swim (and every pixel change, which
    inflates the GIF). Anchoring on units that actually lost HP puts the window on
    the fight; density- or proximity-based anchors kept landing on stacked
    buildings or across the lake.
    """

    damage: dict[int, float] = {}
    position: dict[int, tuple[float, float]] = {}
    previous: dict[int, dict] = {}
    for frame in frames:
        current = {u["id"]: u for u in frame.get("units") or []}
        for uid, unit in current.items():
            position[uid] = (float(unit["x"]), float(unit["y"]))
            before = previous.get(uid)
            if before is not None:
                lost = float(before["hp"]) - float(unit["hp"])
                if lost > 0:
                    damage[uid] = damage.get(uid, 0.0) + lost
        previous = current

    points = [u for frame in frames for u in (frame.get("units") or [])
              if u.get("x") is not None]
    if not points:
        return {"x": 0.0, "y": 0.0, "width": float(map_w), "height": float(map_h)}

    total = sum(damage.values())
    if total > 0:
        anchor = (
            sum(position[uid][0] * loss for uid, loss in damage.items() if uid in position) / total,
            sum(position[uid][1] * loss for uid, loss in damage.items() if uid in position) / total,
        )
    else:
        anchor = (
            sum(u["x"] for u in points) / len(points),
            sum(u["y"] for u in points) / len(points),
        )
    return camera_around(anchor, points, map_w, map_h, out_w, out_h)


# --------------------------------------------------------------------------- #
# frame rendering
# --------------------------------------------------------------------------- #
def render_frame(frame: dict, terrain: Terrain, sprites: SpriteBank,
                 camera: dict, out_w: int, out_h: int,
                 show_hud: bool = True) -> Image.Image:
    """One sandbox-style image for ``frame``."""

    units = frame.get("units") or []
    scale = min(out_w / camera["width"], out_h / camera["height"])
    draw_w = max(1, int(camera["width"] * scale))
    draw_h = max(1, int(camera["height"] * scale))
    origin_x = (out_w - draw_w) // 2
    origin_y = (out_h - draw_h) // 2

    def to_screen(x: float, y: float) -> tuple[float, float]:
        return (origin_x + (x - camera["x"]) * scale, origin_y + (y - camera["y"]) * scale)

    # Background gradient (render.js fills the canvas before anything else).
    image = Image.new("RGB", (out_w, out_h), BG_BOTTOM)
    strip = Image.new("RGB", (1, out_h))
    for row in range(out_h):
        t = row / max(1, out_h - 1)
        strip.putpixel((0, row), (
            int(BG_TOP[0] + (BG_BOTTOM[0] - BG_TOP[0]) * t),
            int(BG_TOP[1] + (BG_BOTTOM[1] - BG_TOP[1]) * t),
            int(BG_TOP[2] + (BG_BOTTOM[2] - BG_TOP[2]) * t),
        ))
    image.paste(strip.resize((out_w, out_h)), (0, 0))

    ground = terrain.render({
        "x": int(camera["x"]),
        "y": int(camera["y"]),
        "width": max(1, int(math.ceil(camera["width"]))),
        "height": max(1, int(math.ceil(camera["height"]))),
    })
    if ground.size != (draw_w, draw_h):
        ground = ground.resize((draw_w, draw_h), Image.LANCZOS)
    image.paste(ground.convert("RGB"), (origin_x, origin_y))

    canvas = Image.new("RGBA", (out_w, out_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    # Lower units overlap upper ones, matching render.js.
    for unit in sorted(units, key=lambda u: u.get("y", 0)):
        own = int(unit.get("team", 1)) == 0
        sx, sy = to_screen(float(unit.get("x", 0)), float(unit.get("y", 0)))
        if not (-80 <= sx <= out_w + 80 and -80 <= sy <= out_h + 80):
            continue

        entry = sprites.entry_for(unit)
        sprite = sprites.image_for(entry)
        body = OWN_BODY if own else ENEMY_BODY
        glow = OWN_GLOW if own else ENEMY_GLOW

        if sprite is not None:
            entry_scale = float((entry or {}).get("scale") or 1.0)
            w = max(4, int(sprite.width * entry_scale * scale))
            h = max(6, int(sprite.height * entry_scale * scale))
            # Contract 2.3: never draw shorter than 10 px.
            if h < 10:
                boost = 10 / h
                w = max(2, int(w * boost))
                h = 10
            tint = Image.alpha_composite(
                sprite, Image.new("RGBA", sprite.size, body + (110,))
            ).resize((w, h), Image.LANCZOS)
            canvas.alpha_composite(tint, (int(sx - w / 2), int(sy - h / 2)))
            body_w, body_h = w, h
        else:
            radius = max(4, int(11 * scale))
            draw.ellipse([sx - radius, sy - radius, sx + radius, sy + radius],
                         fill=body + (255,), outline=glow + (255,), width=2)
            body_w = body_h = radius * 2

        hp = float(unit.get("hp") or 0)
        max_hp = max(1.0, float(unit.get("max_hp") or 1))
        frac = max(0.0, min(1.0, hp / max_hp))
        bar_w = max(14, int(body_w * 0.78))
        bar_h = max(3, int(body_h * 0.10))
        bar_x = sx - bar_w / 2
        bar_y = sy - body_h / 2 - bar_h - 3
        draw.rectangle([bar_x - 1, bar_y - 1, bar_x + bar_w + 1, bar_y + bar_h + 1],
                       fill=(2, 6, 12, 200))
        draw.rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], fill=(40, 46, 56, 255))
        if frac > 0:
            colour = (57, 224, 139) if frac > 0.6 else ((255, 195, 92) if frac > 0.3 else (255, 95, 109))
            draw.rectangle([bar_x, bar_y, bar_x + max(1, bar_w * frac), bar_y + bar_h],
                           fill=colour + (255,))

    image = Image.alpha_composite(image.convert("RGBA"), canvas)

    if show_hud:
        hud = ImageDraw.Draw(image)
        stats = frame.get("stats") or {}
        own_frac = float(stats.get("own_hp_frac", 0) or 0)
        enemy_frac = float(stats.get("enemy_hp_frac", 0) or 0)
        # Plates: the terrain is light desert, so HUD text needs its own backing.
        hud.rectangle([8, 6, 274, 46], fill=(4, 10, 18, 190))
        hud.text((14, 10), f"tick {frame.get('tick')} · t={frame.get('time')}s",
                 font=load_font(14), fill=(200, 228, 250, 255))
        hud.text((14, 29),
                 f"own {stats.get('own_alive')} · enemy {stats.get('enemy_alive')} "
                 f"· {len(units)} units",
                 font=load_font(11), fill=(125, 227, 176, 255))
        plate_h = 34
        hud.rectangle([8, out_h - plate_h - 8, 196, out_h - 8], fill=(4, 10, 18, 190))
        for index, (label, frac, colour) in enumerate(
            (("own", own_frac, OWN_GLOW), ("enemy", enemy_frac, ENEMY_GLOW))
        ):
            yy = out_h - plate_h + 2 + index * 13
            hud.text((16, yy - 1), label, font=load_font(11), fill=(200, 228, 250, 255))
            hud.rectangle([60, yy + 2, 178, yy + 9], fill=(22, 30, 42, 255))
            hud.rectangle([60, yy + 2, 60 + max(0, int(118 * max(0.0, min(1.0, frac)))), yy + 9],
                          fill=colour + (255,))
    return image.convert("RGB")


# --------------------------------------------------------------------------- #
# stream helpers
# --------------------------------------------------------------------------- #
def load_frames(path: pathlib.Path, start: int = 0, count: int | None = None) -> list[dict]:
    frames: list[dict] = []
    seen = 0
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if item.get("type") != "frame":
                continue
            if seen >= start:
                frames.append(item)
                if count is not None and len(frames) >= count:
                    break
            seen += 1
    return frames


def survey_stream(path: pathlib.Path, window: int = 150, top: int = 5) -> None:
    """Print the most active windows of a stream, with a balance filter."""

    frames = load_frames(path)
    print(f"stream : {path.parent.name}")
    print(f"frames : {len(frames)}")
    if len(frames) < window + 2:
        print("too short to survey")
        return

    def mapping(frame: dict) -> dict[int, dict]:
        return {u["id"]: u for u in frame.get("units") or []}

    motion = [0.0] * len(frames)
    damage = [0.0] * len(frames)
    prev = None
    for index, frame in enumerate(frames):
        current = mapping(frame)
        if prev is not None:
            moved = hurt = 0.0
            for uid, unit in current.items():
                before = prev.get(uid)
                if before is None:
                    continue
                dx = unit["x"] - before["x"]
                dy = unit["y"] - before["y"]
                moved += (dx * dx + dy * dy) ** 0.5
                hurt += max(0.0, before["hp"] - unit["hp"])
            motion[index] = moved
            damage[index] = hurt
        prev = current

    scored = []
    for start in range(0, len(frames) - window, max(1, window // 15)):
        seg = frames[start:start + window]
        stats = seg[-1].get("stats") or {}
        own = float(stats.get("own_alive") or 0)
        enemy = float(stats.get("enemy_alive") or 0)
        if min(own, enemy) < 4:
            continue
        balance = min(own, enemy) / max(1.0, max(own, enemy))
        move = sum(motion[start:start + window])
        dmg = sum(damage[start:start + window])
        units = sum(len(f.get("units") or []) for f in seg) / window
        scored.append((move + dmg * 3 + balance * 120 + units * 2, start, seg[-1].get("time"),
                       own, enemy, units, move, dmg))
    scored.sort(reverse=True)

    print(f"\n{'start':>7}{'t(s)':>10}{'own':>5}{'enemy':>7}{'units':>7}"
          f"{'motion':>9}{'damage':>8}")
    for score, start, time_s, own, enemy, units, move, dmg in scored[:top]:
        print(f"{start:>7}{time_s:>10.1f}{own:>5.0f}{enemy:>7.0f}"
              f"{units:>7.0f}{move:>9.0f}{dmg:>8.0f}")
    if scored:
        print(f"\nsuggested: --start {scored[0][1]} --frames {window}")


def find_ffmpeg() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    for candidate in (r"E:\EVCapture\ffmpeg.exe", r"C:\ffmpeg\bin\ffmpeg.exe"):
        if pathlib.Path(candidate).is_file():
            return candidate
    return None


def encode_gif(images: list[Image.Image], out_path: pathlib.Path,
               fps: int, colors: int) -> str:
    """Write a GIF, using ffmpeg's palettegen pipeline when available.

    Per-frame Pillow quantisation is measurably poor here: the same 100-frame clip
    came out at 7 MB with Pillow and 556 KB through ffmpeg (one palette for the
    whole clip + sierra2_4a dithering). Fall back only when ffmpeg is missing.
    """

    ffmpeg = find_ffmpeg()
    if ffmpeg:
        with tempfile.TemporaryDirectory(prefix="ironzero-gif-") as tmp:
            tmp_dir = pathlib.Path(tmp)
            for index, image in enumerate(images):
                image.save(tmp_dir / f"f{index:05d}.png")
            palette = tmp_dir / "palette.png"
            source = str(tmp_dir / "f%05d.png")
            first = subprocess.run(
                [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps),
                 "-i", source, "-vf", f"palettegen=max_colors={colors}", str(palette)],
                capture_output=True, text=True,
            )
            if first.returncode == 0 and palette.is_file():
                second = subprocess.run(
                    [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps),
                     "-i", source, "-i", str(palette),
                     "-lavfi", "paletteuse=dither=sierra2_4a",
                     "-loop", "0", str(out_path)],
                    capture_output=True, text=True,
                )
                if second.returncode == 0 and out_path.is_file():
                    return "ffmpeg"
            print("ffmpeg GIF encode failed; falling back to Pillow", file=sys.stderr)

    paletted = [
        image.quantize(colors=colors, method=Image.MEDIANCUT, dither=Image.FLOYDSTEINBERG)
        for image in images
    ]
    paletted[0].save(
        out_path, save_all=True, append_images=paletted[1:],
        duration=max(20, int(round(1000 / max(1, fps)))),
        loop=0, optimize=True, disposal=2,
    )
    return "Pillow"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render sandbox-style images/GIFs from a recorded frame stream"
    )
    parser.add_argument("--stream", required=True,
                        help="job directory name under runs/platform")
    parser.add_argument("--runs-root", default=str(REPO / "runs" / "platform"))
    parser.add_argument("--backend", default=DEFAULT_BACKEND,
                        help="platform backend serving /api/maps and /api/assets")
    parser.add_argument("--map", default="[p2]Lake (2p)", help="map name to render")
    parser.add_argument("--start", type=int, default=0, help="first frame index")
    parser.add_argument("--frames", type=int, default=150, help="frames to read")
    parser.add_argument("--stride", type=int, default=1, help="keep every Nth frame")
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=450)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--colors", type=int, default=96)
    parser.add_argument("--still", default="", help="write a single PNG of the first frame")
    parser.add_argument("--gif", default="", help="write an animated GIF")
    parser.add_argument("--survey", action="store_true",
                        help="print the most active windows and exit")
    parser.add_argument("--camera", default="auto",
                        choices=["auto", "clip", "full"],
                        help="auto=damage-anchored fixed window; full=whole map")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    runs_root = pathlib.Path(args.runs_root)
    stream_path = runs_root / args.stream / "frames.jsonl"
    if not stream_path.is_file():
        print(f"no such stream: {stream_path}", file=sys.stderr)
        return 2

    if args.survey:
        survey_stream(stream_path)
        return 0

    if not args.still and not args.gif:
        print("nothing to do: pass --still and/or --gif (or --survey)", file=sys.stderr)
        return 2

    print(f"stream : {args.stream} ({stream_path.stat().st_size / 1024 / 1024:.1f} MB)")
    frames = load_frames(stream_path, start=args.start, count=args.frames)
    if not frames:
        print("no frames in that range", file=sys.stderr)
        return 2
    kept = frames[:: max(1, args.stride)]
    print(f"clip   : {len(kept)} frames (start={args.start}, stride={args.stride})")

    maps = fetch_json(args.backend, "/api/maps")
    entry = next((m for m in maps.get("maps") or [] if m.get("name") == args.map), None)
    if entry is None:
        print(f"map not found: {args.map}", file=sys.stderr)
        return 2
    detail = fetch_json(args.backend, f"/api/maps/{entry['id']}")
    terrain = Terrain(args.backend, detail)
    sprites = SpriteBank(args.backend, maps.get("unitSprites") or {})
    print(f"assets : {args.map} {detail['width_px']}x{detail['height_px']}, "
          f"{len(terrain.atlases)} tilesets, {len(sprites.by_type)} unit types")

    map_w, map_h = int(detail["width_px"]), int(detail["height_px"])
    if args.camera == "full":
        camera = {"x": 0.0, "y": 0.0, "width": float(map_w), "height": float(map_h)}
    else:
        camera = camera_for_clip(kept, map_w, map_h, args.width, args.height)
    print(f"camera : x={camera['x']:.0f} y={camera['y']:.0f} "
          f"{camera['width']:.0f}x{camera['height']:.0f} map px")

    if args.still:
        target = pathlib.Path(args.still)
        target.parent.mkdir(parents=True, exist_ok=True)
        render_frame(kept[0], terrain, sprites, camera, args.width, args.height).save(target)
        print(f"still  : {target}")

    if args.gif:
        target = pathlib.Path(args.gif)
        target.parent.mkdir(parents=True, exist_ok=True)
        images = [
            render_frame(frame, terrain, sprites, camera, args.width, args.height)
            for frame in kept
        ]
        encoder = encode_gif(images, target, args.fps, args.colors)
        print(f"gif    : {target}  {target.stat().st_size / 1024:.0f} KB  "
              f"({len(images)} frames, {args.width}x{args.height}, {args.fps} fps, {encoder})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
