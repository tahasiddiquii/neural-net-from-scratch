"""Stochastic gradient descent with classical momentum."""

from __future__ import annotations

import numpy as np


class SGD:
    def __init__(self, lr: float = 0.3, momentum: float = 0.9) -> None:
        self.lr = lr
        self.momentum = momentum
        self._velocity: dict[int, np.ndarray] = {}

    def step(self, net) -> None:
        for param, grad in net.params_and_grads():
            key = id(param)
            velocity = self._velocity.get(key)
            if velocity is None:
                velocity = np.zeros_like(param)
            velocity = self.momentum * velocity - self.lr * grad
            self._velocity[key] = velocity
            param += velocity  # in-place: the layer holds the same array
