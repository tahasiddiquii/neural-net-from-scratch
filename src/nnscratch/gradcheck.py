"""Gradient checking — the proof that backprop is implemented correctly.

For a handful of parameters, compare the analytical gradient (from ``backward``) to a
numerical one from central finite differences. If they don't match to ~1e-6, the backprop
math is wrong. This is the single most valuable habit when writing a network by hand.
"""

from __future__ import annotations

import numpy as np

from nnscratch.losses import softmax_cross_entropy
from nnscratch.network import MLP


def gradient_check(
    net: MLP, x: np.ndarray, y: np.ndarray, eps: float = 1e-5, samples: int = 8, seed: int = 0
) -> float:
    """Return the max relative error between analytical and numerical gradients."""
    # Populate analytical gradients.
    _, dlogits = softmax_cross_entropy(net.forward(x), y)
    net.backward(dlogits)

    rng = np.random.default_rng(seed)
    max_rel = 0.0
    for param, grad in net.params_and_grads():
        flat = param.ravel()  # view — writing to it updates the parameter in place
        gflat = grad.ravel()
        for idx in rng.choice(flat.size, size=min(samples, flat.size), replace=False):
            original = flat[idx]
            flat[idx] = original + eps
            loss_plus = softmax_cross_entropy(net.forward(x), y)[0]
            flat[idx] = original - eps
            loss_minus = softmax_cross_entropy(net.forward(x), y)[0]
            flat[idx] = original

            numerical = (loss_plus - loss_minus) / (2 * eps)
            analytical = gflat[idx]
            denom = max(1e-8, abs(numerical) + abs(analytical))
            max_rel = max(max_rel, abs(numerical - analytical) / denom)
    return max_rel
