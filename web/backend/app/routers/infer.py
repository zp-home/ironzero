# SPDX-License-Identifier: Apache-2.0
"""Inference jobs: start a recording worker, poll status, stream frames."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from ... import maps as gamemaps
from ..jobs import _job_status, _new_job_dir, _python, _read_meta, _spawn
from ..jsonl_index import MAX_WINDOW_RECORDS, jsonl_index
from ..models import InferRequest
from ..paths import RUNS_ROOT, _resolve_inside

router = APIRouter()


@router.post("/api/infer")
def start_infer(request: InferRequest) -> dict[str, Any]:
    checkpoint = request.checkpoint.strip()
    if checkpoint and checkpoint.lower() not in {"random", "none"}:
        path = Path(checkpoint).resolve()
        if not path.is_file():
            raise HTTPException(status_code=400, detail=f"checkpoint not found: {checkpoint}")
        checkpoint = str(path)
    else:
        checkpoint = "random"

    sim_mode = request.sim_mode
    map_id = request.map_id.strip()
    if sim_mode == "terrain" and not map_id:
        raise HTTPException(status_code=400, detail="地形模拟需要先选择一张地图")
    if map_id and sim_mode == "arena":
        sim_mode = "terrain"

    job_id, job_dir = _new_job_dir("infer")
    replays = job_dir / "replays"
    config = request.model_dump()
    config["checkpoint"] = checkpoint
    config["sim_mode"] = sim_mode
    config["name"] = (
        f"推理 {request.episodes} 局 × {request.opponent}"
        + (" · 地形" if sim_mode == "terrain" else " · 竞技场")
    )
    command = [
        _python(),
        "-m",
        "ironzero.training.record",
        "--checkpoint",
        checkpoint,
        "--episodes",
        str(request.episodes),
        "--opponent",
        request.opponent,
        "--seed",
        str(request.seed),
        "--mode",
        request.mode,
        "--max-decisions",
        str(request.max_decisions),
        "--device",
        "cpu",
        "--out-dir",
        str(replays),
        "--stream",
        str(job_dir / "frames.jsonl"),
        "--prefix",
        job_id,
    ]
    if sim_mode == "terrain":
        try:
            map_payload = gamemaps.load_map(map_id)
        except gamemaps.MapError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        blocked = gamemaps.blocked_mask(map_payload)
        map_json = {
            "id": map_payload["id"],
            "name": map_payload["name"],
            "width": map_payload["width"],
            "height": map_payload["height"],
            "tile_width": map_payload["tile_width"],
            "tile_height": map_payload["tile_height"],
            "width_px": map_payload["width_px"],
            "height_px": map_payload["height_px"],
            "blocked": blocked,
        }
        map_path = job_dir / "map.json"
        map_path.write_text(json.dumps(map_json, ensure_ascii=False), encoding="utf-8")
        config["map_id"] = map_id
        config["map_name"] = map_payload["name"]
        command.extend(["--sim-mode", "terrain", "--map-json", str(map_path)])
    else:
        # Arena mode keeps the historical worker command (no sim flags needed).
        pass
    if map_id:
        command.extend(["--map-id", map_id])
    if request.fog:
        command.extend(["--fog", "--vision-radius", str(request.vision_radius)])
        config["fog"] = True
        config["vision_radius"] = request.vision_radius
    else:
        config["fog"] = False
    _spawn(job_id, job_dir, "infer", config, command)
    return {"id": job_id, "status": "running", "dir": str(job_dir), "sim_mode": sim_mode}


@router.get("/api/infer/{infer_id}")
def get_infer(infer_id: str) -> dict[str, Any]:
    job_dir = _resolve_inside(RUNS_ROOT, infer_id)
    meta = _read_meta(job_dir)
    if not meta:
        raise HTTPException(status_code=404, detail="inference job not found")
    # Index-backed: counting frames and fetching the last one must not parse the
    # whole stream. This endpoint is polled every 700 ms, and a long episode's
    # frames.jsonl reaches 100+ MB, so a full parse here (the previous behaviour)
    # stalled every other request in the process.
    stream = jsonl_index(job_dir / "frames.jsonl")
    episode_ends = stream.read_kind("episode_end")
    return {
        **meta,
        "status": _job_status(infer_id, job_dir),
        "frames_total": stream.kind_count("frame"),
        "latest_frame": stream.read_last("frame"),
        "results": episode_ends,
        "episodes_done": len(episode_ends),
        "done": stream.has("done"),
    }


@router.get("/api/infer/{infer_id}/frames")
def get_infer_frames(
    infer_id: str,
    after: int = Query(default=0, ge=0),
    limit: int | None = Query(
        default=None,
        ge=0,
        le=MAX_WINDOW_RECORDS,
        description=(
            "Records to return. Omit for the whole tail (unchanged behaviour); "
            "0 for a server-chosen bounded chunk; N to cap at N."
        ),
    ),
) -> dict[str, Any]:
    job_dir = _resolve_inside(RUNS_ROOT, infer_id)
    # ``after`` is a record index (not a byte offset) and ``next`` is always the
    # absolute index to ask for next time, so a capped window needs no special
    # handling on the client: it just calls again and catches up. With no
    # ``limit`` the response is the full tail, which is what the replay loader
    # relies on.
    frames, next_cursor = jsonl_index(job_dir / "frames.jsonl").read_window(after, limit)
    return {
        "next": next_cursor,
        "frames": frames,
        "status": _job_status(infer_id, job_dir),
    }
