"""The load-bearing test: backprop matches numerical gradients."""

from __future__ import annotations

from nnscratch.data import make_spiral, train_test_split
from nnscratch.gradcheck import gradient_check
from nnscratch.network import MLP


def test_backprop_matches_finite_differences():
    x, y = make_spiral(seed=1)
    # Use a shuffled (class-mixed) batch: an all-one-class batch creates degenerate
    # ReLU kinks that make finite-difference checks unreliable.
    x_train, y_train, _, _ = train_test_split(x, y, seed=1)
    rel = gradient_check(MLP((2, 32, 3), seed=1), x_train[:40], y_train[:40])
    assert rel < 1e-4, f"gradient mismatch: {rel:.2e}"
