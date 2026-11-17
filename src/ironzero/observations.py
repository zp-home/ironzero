# SPDX-License-Identifier: Apache-2.0
"""Compatibility shim: the implementation now lives in ``ironzero.observation``.

Every public name that used to be defined in this module is re-exported below so
``from ironzero.observations import ...`` keeps working unchanged. Nothing is
implemented here.
"""

from __future__ import annotations

from .observation import (  # noqa: F401
    GROUP_DIM,
    MULTI_MASK_KEYS,
    build_masks,
    build_masks_for_config,
    build_masks_multi,
    encode_observation,
    flat_mask_vector,
    masks_as_lists,
    multi_mask_keys,
    terrain_passable_channel,
)

__all__ = [
    "GROUP_DIM",
    "MULTI_MASK_KEYS",
    "build_masks",
    "build_masks_for_config",
    "build_masks_multi",
    "encode_observation",
    "flat_mask_vector",
    "masks_as_lists",
    "multi_mask_keys",
    "terrain_passable_channel",
]
