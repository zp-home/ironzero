# SPDX-License-Identifier: Apache-2.0
"""Minimal Gymnasium-like space descriptors for :class:`RwppDuelEnv`.

These are shape/dtype descriptors only: the training and evaluation code never
samples from them, they exist so the environment exposes the usual
``observation_space`` / ``action_space`` attributes.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class _BoxSpace:
    def __init__(self, shape: tuple[int, ...], dtype: Any = np.float32) -> None:
        self.shape = shape
        self.dtype = dtype

    def sample(self) -> np.ndarray:  # pragma: no cover - diagnostics only
        return np.zeros(self.shape, dtype=self.dtype)

    def contains(self, value: np.ndarray) -> bool:
        return getattr(value, "shape", None) == self.shape


class _MultiDiscreteSpace:
    def __init__(self, nvec: tuple[int, ...]) -> None:
        self.nvec = np.asarray(nvec, dtype=np.int64)

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self.nvec),)

    def sample(self) -> np.ndarray:
        return np.array([np.random.randint(0, int(n)) for n in self.nvec], dtype=np.int64)

    def contains(self, value: np.ndarray) -> bool:
        return (
            isinstance(value, np.ndarray)
            and value.shape == self.shape
            and np.all(value >= 0)
            and np.all(value < self.nvec)
        )


__all__ = ["_BoxSpace", "_MultiDiscreteSpace"]
