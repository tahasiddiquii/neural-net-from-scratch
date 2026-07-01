"""Softmax + cross-entropy."""

from __future__ import annotations

import numpy as np

from nnscratch.losses import softmax, softmax_cross_entropy


def test_softmax_normalizes_rows():
    p = softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
    assert np.allclose(p.sum(axis=1), 1.0)


def test_cross_entropy_loss_and_gradient():
    logits = np.array([[2.0, 0.0, 0.0]])
    loss, grad = softmax_cross_entropy(logits, np.array([0]))
    assert loss > 0
    assert grad.shape == (1, 3)
    # increasing the true-class logit should reduce loss -> negative gradient there
    assert grad[0, 0] < 0
