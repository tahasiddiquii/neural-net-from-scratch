"""Layers — each owns its parameters and knows its own local gradient.

A layer implements ``forward`` (cache what backprop needs) and ``backward`` (given the
gradient of the loss w.r.t. its output, return the gradient w.r.t. its input and stash the
gradients w.r.t. its parameters). Composing these *is* backpropagation.
"""

from __future__ import annotations

import numpy as np


class Linear:
    """Affine layer: y = x @ W + b."""

    def __init__(self, in_dim: int, out_dim: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        # He initialization keeps activation variance stable through ReLU layers.
        self.W = rng.normal(0.0, np.sqrt(2.0 / in_dim), (in_dim, out_dim))
        self.b = np.zeros(out_dim)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.W + self.b

    def backward(self, dout: np.ndarray) -> np.ndarray:
        self.dW = self._x.T @ dout
        self.db = dout.sum(axis=0)
        return dout @ self.W.T

    def params(self) -> list[np.ndarray]:
        return [self.W, self.b]

    def grads(self) -> list[np.ndarray]:
        return [self.dW, self.db]


class ReLU:
    """Elementwise max(0, x)."""

    def __init__(self) -> None:
        self._mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = x > 0
        return x * self._mask

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self._mask

    def params(self) -> list[np.ndarray]:
        return []

    def grads(self) -> list[np.ndarray]:
        return []
