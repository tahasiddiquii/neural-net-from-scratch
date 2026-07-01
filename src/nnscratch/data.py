"""A spiral dataset — non-linearly separable, so a linear model can't solve it.

Three interleaved spiral arms (the classic CS231n toy problem). An MLP with a hidden layer
learns the curved decision boundaries; logistic regression cannot. Generated from a fixed
seed, so training is reproducible.
"""

from __future__ import annotations

import numpy as np


def make_spiral(
    points_per_class: int = 120, classes: int = 3, noise: float = 0.2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = points_per_class
    x = np.zeros((n * classes, 2))
    y = np.zeros(n * classes, dtype=int)
    for j in range(classes):
        radius = np.linspace(0.0, 1.0, n)
        theta = np.linspace(j * 4, (j + 1) * 4, n) + rng.normal(0, noise, n)
        x[j * n : (j + 1) * n] = np.c_[radius * np.sin(theta), radius * np.cos(theta)]
        y[j * n : (j + 1) * n] = j
    return x, y


def train_test_split(
    x: np.ndarray, y: np.ndarray, test_frac: float = 0.2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 100)
    idx = rng.permutation(len(y))
    cut = int(len(y) * (1 - test_frac))
    train_idx, test_idx = idx[:cut], idx[cut:]
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]
