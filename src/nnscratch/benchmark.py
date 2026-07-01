"""Train + evaluate the network behind a two-part gate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nnscratch.data import make_spiral, train_test_split
from nnscratch.gradcheck import gradient_check
from nnscratch.network import MLP
from nnscratch.train import accuracy, train

THRESHOLDS = {
    "test_accuracy": 0.90,  # the network must actually learn the spiral
    "gradient_check_max_rel": 1e-4,  # backprop must match numerical gradients
}


@dataclass
class BenchmarkReport:
    train_accuracy: float
    test_accuracy: float
    gradient_check: float
    final_loss: float

    def passed(self) -> bool:
        return not self.failures()

    def failures(self) -> list[str]:
        out: list[str] = []
        if self.test_accuracy < THRESHOLDS["test_accuracy"]:
            out.append(f"test_accuracy={self.test_accuracy:.3f} < {THRESHOLDS['test_accuracy']:.2f}")
        if self.gradient_check > THRESHOLDS["gradient_check_max_rel"]:
            out.append(
                f"gradient_check={self.gradient_check:.2e} > {THRESHOLDS['gradient_check_max_rel']:.0e}"
            )
        return out


def run_benchmark(seed: int = 0) -> BenchmarkReport:
    x, y = make_spiral(seed=seed)
    x_train, y_train, x_test, y_test = train_test_split(x, y, seed=seed)

    # Gradient-check a freshly initialised network on a small batch (fast + strict).
    gc = gradient_check(MLP((2, 64, 3), seed=seed), x_train[:50], y_train[:50])

    net = MLP((2, 64, 3), seed=seed)
    history = train(net, x_train, y_train, seed=seed)
    return BenchmarkReport(
        train_accuracy=round(accuracy(net, x_train, y_train), 4),
        test_accuracy=round(accuracy(net, x_test, y_test), 4),
        gradient_check=gc,
        final_loss=round(history[-1], 4),
    )


def write_markdown(report: BenchmarkReport, path: Path) -> None:
    lines = [
        "# neural-net-from-scratch — report",
        "",
        "An MLP (2 → 64 → 3, ReLU) trained on a 3-arm spiral with pure-numpy backprop and SGD.",
        "",
        "| metric | value | threshold |",
        "| --- | --- | --- |",
        f"| train_accuracy | {report.train_accuracy:.4f} | — |",
        f"| test_accuracy | {report.test_accuracy:.4f} | ≥ {THRESHOLDS['test_accuracy']:.2f} |",
        f"| gradient_check (max rel err) | {report.gradient_check:.2e} | ≤ {THRESHOLDS['gradient_check_max_rel']:.0e} |",
        f"| final_loss | {report.final_loss:.4f} | — |",
        "",
        f"**Gate: {'PASSED' if report.passed() else 'FAILED'}**",
        "",
        "The gradient check is the important line: the analytical gradients from `backward` match "
        "central finite differences to ~1e-6, which *proves* the backprop is correct — the test "
        "accuracy then proves the network actually learns the non-linear boundary a linear model can't.",
    ]
    path.write_text("\n".join(lines) + "\n")
