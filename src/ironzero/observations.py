# SPDX-License-Identifier: Apache-2.0
"""Compatibility shim: the implementation now lives in ``ironzero.observation``.

Every public name that used to be defined in this module is re-exported below so
``from ironzero.observations import ...`` keeps working unchanged. Nothing is
implemented here.
"""

from __future__ import annotations

from .observation import (  # noqa: F401
    GROUP_DIM,
    build_masks,
    encode_observation,
    flat_mask_vector,
    masks_as_lists,
    terrain_passable_channel,
)

__all__ = [
    "GROUP_DIM",
    "build_masks",
    "encode_observation",
    "flat_mask_vector",
    "masks_as_lists",
    "terrain_passable_channel",
]
