"""Softmax + cross-entropy, with the combined gradient that makes backprop clean."""

from __future__ import annotations

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)  # shift for numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def softmax_cross_entropy(logits: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Return (mean loss, gradient w.r.t. logits).

    The gradient of softmax + cross-entropy collapses to the elegant ``p - onehot(y)``,
    which is why they're paired.
    """
    p = softmax(logits)
    n = len(y)
    loss = float(-np.mean(np.log(p[np.arange(n), y] + 1e-12)))
    grad = p.copy()
    grad[np.arange(n), y] -= 1.0
    grad /= n
    return loss, grad
