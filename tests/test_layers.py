"""Layer forward/backward shapes and behavior."""

from __future__ import annotations

import numpy as np

from nnscratch.layers import Linear, ReLU


def test_linear_forward_and_backward_shapes():
    lin = Linear(3, 2, seed=0)
    x = np.random.default_rng(0).normal(0, 1, (4, 3))
    out = lin.forward(x)
    assert out.shape == (4, 2)
    dx = lin.backward(np.ones((4, 2)))
    assert dx.shape == (4, 3)
    assert lin.dW.shape == (3, 2)
    assert lin.db.shape == (2,)


def test_relu_gates_negatives():
    relu = ReLU()
    assert np.allclose(relu.forward(np.array([[-1.0, 2.0]])), [[0.0, 2.0]])
    assert np.allclose(relu.backward(np.ones((1, 2))), [[0.0, 1.0]])
