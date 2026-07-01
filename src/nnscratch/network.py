"""The network — a stack of layers with a shared forward/backward interface."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from nnscratch.layers import Linear, ReLU


class MLP:
    def __init__(self, sizes: tuple[int, ...] = (2, 64, 3), seed: int = 0) -> None:
        self.layers: list[object] = []
        for i in range(len(sizes) - 1):
            self.layers.append(Linear(sizes[i], sizes[i + 1], seed=seed + i))
            if i < len(sizes) - 2:  # no activation after the final (logit) layer
                self.layers.append(ReLU())

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout: np.ndarray) -> None:
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

    def params_and_grads(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for layer in self.layers:
            yield from zip(layer.params(), layer.grads(), strict=True)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x).argmax(axis=1)
