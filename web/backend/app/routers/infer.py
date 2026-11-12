# SPDX-License-Identifier: Apache-2.0
"""Inference jobs: start a recording worker, poll status, stream frames."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from ... import maps as gamemaps
from ..jobs import _job_status, _new_job_dir, _python, _read_jsonl, _read_meta, _spawn
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
    stream = _read_jsonl(job_dir / "frames.jsonl")
    frames = [item for item in stream if item.get("type") == "frame"]
    episode_ends = [item for item in stream if item.get("type") == "episode_end"]
    done = any(item.get("type") == "done" for item in stream)
    return {
        **meta,
        "status": _job_status(infer_id, job_dir),
        "frames_total": len(frames),
        "latest_frame": frames[-1] if frames else None,
        "results": episode_ends,
        "episodes_done": len(episode_ends),
        "done": done,
    }


@router.get("/api/infer/{infer_id}/frames")
def get_infer_frames(
    infer_id: str, after: int = Query(default=0, ge=0)
) -> dict[str, Any]:
    job_dir = _resolve_inside(RUNS_ROOT, infer_id)
    stream = _read_jsonl(job_dir / "frames.jsonl")
    pending = stream[after:]
    return {
        "next": after + len(pending),
        "frames": pending,
        "status": _job_status(infer_id, job_dir),
    }
