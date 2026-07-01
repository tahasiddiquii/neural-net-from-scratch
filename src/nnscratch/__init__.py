"""neural-net-from-scratch — an MLP with hand-derived backprop in pure numpy.

Layers, activations, softmax cross-entropy, and SGD with momentum, trained on a
non-linearly-separable spiral. A numerical gradient check proves the analytical gradients
are correct, and an accuracy gate proves the network actually learns. No DL framework.
"""

from __future__ import annotations

from nnscratch.benchmark import THRESHOLDS, BenchmarkReport, run_benchmark
from nnscratch.data import make_spiral, train_test_split
from nnscratch.gradcheck import gradient_check
from nnscratch.layers import Linear, ReLU
from nnscratch.losses import softmax, softmax_cross_entropy
from nnscratch.network import MLP
from nnscratch.optim import SGD
from nnscratch.train import accuracy, train

__version__ = "0.1.0"

__all__ = [
    "MLP",
    "SGD",
    "THRESHOLDS",
    "BenchmarkReport",
    "Linear",
    "ReLU",
    "accuracy",
    "gradient_check",
    "make_spiral",
    "run_benchmark",
    "softmax",
    "softmax_cross_entropy",
    "train",
    "train_test_split",
    "__version__",
]
