"""Training loop + accuracy."""

from __future__ import annotations

import numpy as np

from nnscratch.losses import softmax_cross_entropy
from nnscratch.network import MLP
from nnscratch.optim import SGD


def accuracy(net: MLP, x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(net.predict(x) == y))


def train(
    net: MLP,
    x: np.ndarray,
    y: np.ndarray,
    epochs: int = 300,
    batch_size: int = 32,
    lr: float = 0.3,
    momentum: float = 0.9,
    seed: int = 0,
) -> list[float]:
    """Mini-batch SGD. Returns the per-epoch loss history."""
    opt = SGD(lr, momentum)
    rng = np.random.default_rng(seed)
    n = len(y)
    history: list[float] = []
    for _ in range(epochs):
        order = rng.permutation(n)
        last_loss = 0.0
        for start in range(0, n, batch_size):
            batch = order[start : start + batch_size]
            loss, dlogits = softmax_cross_entropy(net.forward(x[batch]), y[batch])
            net.backward(dlogits)
            opt.step(net)
            last_loss = loss
        history.append(last_loss)
    return history
