"""Training actually learns + the end-to-end gate."""

from __future__ import annotations

from nnscratch.benchmark import run_benchmark
from nnscratch.data import make_spiral, train_test_split
from nnscratch.network import MLP
from nnscratch.train import accuracy, train


def test_training_reduces_loss_and_generalizes():
    x, y = make_spiral(seed=2)
    x_train, y_train, x_test, y_test = train_test_split(x, y, seed=2)
    net = MLP((2, 64, 3), seed=2)
    history = train(net, x_train, y_train, epochs=200, seed=2)
    assert history[-1] < history[0]
    assert accuracy(net, x_test, y_test) > 0.85


def test_gate_passes_and_is_verified():
    report = run_benchmark()
    assert report.passed(), report.failures()
    assert report.test_accuracy >= 0.90
    assert report.gradient_check <= 1e-4
