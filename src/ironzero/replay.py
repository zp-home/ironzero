# SPDX-License-Identifier: Apache-2.0
"""Replay recording for micro-duel episodes.

A replay is a plain JSON document with a fixed frame schema so the dashboard,
the video recorder and future tools can all read the same data::

    {
      "schema": "ironzero.replay/1",
      "meta": {...},
      "frames": [
         {"episode": 0, "tick": 0, "time": 0.0, "units": [...],
          "events": [...], "stats": {...}, "result": null},
         ...
      ]
    }

Frames are recorded once per decision window (0.5 s of game time), which is
enough for smooth playback with client-side interpolation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .micro_config import MicroDuelConfig

REPLAY_SCHEMA = "ironzero.replay/1"


def build_meta(
    *,
    checkpoint: str | None,
    opponent: str,
    seed: int,
    config: MicroDuelConfig,
    episodes: int,
    mode: str,
    created_at: str,
) -> dict[str, Any]:
    return {
        "schema": REPLAY_SCHEMA,
        "checkpoint": checkpoint or "random",
        "opponent": opponent,
        "seed": seed,
        "episodes": episodes,
        "mode": mode,
        "created_at": created_at,
        "map": {"width": config.map_width, "height": config.map_height},
        "tick_rate": config.tick_rate,
        "decision_interval": config.decision_interval,
        "units_per_team": config.units_per_team,
        "groups": config.groups,
    }


def episode_meta(
    *, index: int, seed: int, result: str | None, total_reward: float, decisions: int
) -> dict[str, Any]:
    return {
        "episode": index,
        "seed": seed,
        "result": result,
        "return": round(total_reward, 4),
        "decisions": decisions,
    }


def collect_units(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten own + visible enemy entities into one render-ready unit list."""

    units: list[dict[str, Any]] = []
    for entity in state.get("own_entities", []):
        units.append(_unit_record(entity))
    for entity in state.get("visible_enemy_entities", []):
        units.append(_unit_record(entity))
    return units


def _unit_record(entity: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": int(entity.get("entity_id", 0)),
        "team": int(entity.get("team_id", 0)),
        "group": int(entity.get("group_id", 0)),
        "x": round(float(entity.get("x", 0.0)), 3),
        "y": round(float(entity.get("y", 0.0)), 3),
        "hp": round(float(entity.get("hp") or 0.0), 3),
        "max_hp": round(float(entity.get("max_hp") or 1.0), 3),
        "order": str(entity.get("order_type", "idle")),
        "target": entity.get("attack_target_id"),
    }


def build_frame(
    state: dict[str, Any],
    *,
    episode: int,
    decision: int,
    events: Iterable[dict[str, Any]] | None = None,
    config: MicroDuelConfig,
) -> dict[str, Any]:
    own = state.get("own_entities", [])
    enemies = state.get("visible_enemy_entities", [])
    total_hp = max(1.0, float(config.units_per_team * config.unit_hp))
    return {
        "episode": episode,
        "decision": decision,
        "tick": int(state.get("tick", 0)),
        "time": round(float(state.get("tick", 0)) / config.tick_rate, 3),
        "reward": round(float(state.get("reward", 0.0)), 4),
        "units": collect_units(state),
        "events": list(events or []),
        "stats": {
            "own_hp_frac": round(sum(float(u.get("hp") or 0.0) for u in own) / total_hp, 4),
            "enemy_hp_frac": round(sum(float(u.get("hp") or 0.0) for u in enemies) / total_hp, 4),
            "own_alive": len(own),
            "enemy_alive": len(enemies),
        },
        "result": state.get("result"),
    }


def write_replay(
    path: str | Path,
    *,
    meta: dict[str, Any],
    frames: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": REPLAY_SCHEMA, "meta": meta, "episodes": episodes, "frames": frames}
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return path


def load_replay(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "REPLAY_SCHEMA",
    "build_frame",
    "build_meta",
    "collect_units",
    "episode_meta",
    "load_replay",
    "write_replay",
]
