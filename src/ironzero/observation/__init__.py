# SPDX-License-Identifier: Apache-2.0
"""Observation encoding and action masks for the micro-duel environment.

The encoder turns the raw observation payload produced by the game-side
adapter (mock engine or a future RWPP bridge) into a flat ``float32`` vector,
and derives the per-head action masks used by the policy. Both live in this
package so the mask layout and the observation layout stay aligned with
``MicroDuelConfig``.

Package layout:

============================  ====================================================
``encoding.py``                :func:`encode_observation`: payload -> flat vector.
``features.py``                per-entity / per-cell features the encoder writes.
``terrain_channel.py``         static terrain passability prior (channel 3).
``masks.py``                   :func:`build_masks` and its serializers.
============================  ====================================================

The historical module path ``ironzero.observations`` still works: it is a thin
re-export shim over this package.
"""

from __future__ import annotations

from .encoding import encode_observation
from .features import GROUP_DIM, _cell, _entity_features, _nearest, _unit_group
from .masks import build_masks, flat_mask_vector, masks_as_lists
from .terrain_channel import (
    REPO_ROOT,
    _downsample_passable,
    _resolve_terrain_path,
    _terrain_passable_channel,
    terrain_passable_channel,
)

__all__ = [
    "GROUP_DIM",
    "REPO_ROOT",
    "_cell",
    "_downsample_passable",
    "_entity_features",
    "_nearest",
    "_resolve_terrain_path",
    "_terrain_passable_channel",
    "_unit_group",
    "build_masks",
    "encode_observation",
    "flat_mask_vector",
    "masks_as_lists",
    "terrain_passable_channel",
]
