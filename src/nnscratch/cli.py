"""Command-line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from nnscratch.benchmark import THRESHOLDS, run_benchmark, write_markdown
from nnscratch.data import make_spiral, train_test_split
from nnscratch.gradcheck import gradient_check
from nnscratch.network import MLP

console = Console()


def _cmd_benchmark(args) -> int:
    report = run_benchmark()
    table = Table(title="neural-net-from-scratch")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_column("threshold", justify="right")
    table.add_column("", justify="center")
    table.add_row("train_accuracy", f"{report.train_accuracy:.4f}", "—", "")
    table.add_row(
        "test_accuracy",
        f"{report.test_accuracy:.4f}",
        f"{THRESHOLDS['test_accuracy']:.2f}",
        "✅" if report.test_accuracy >= THRESHOLDS["test_accuracy"] else "❌",
    )
    table.add_row(
        "gradient_check",
        f"{report.gradient_check:.2e}",
        f"{THRESHOLDS['gradient_check_max_rel']:.0e}",
        "✅" if report.gradient_check <= THRESHOLDS["gradient_check_max_rel"] else "❌",
    )
    table.add_row("final_loss", f"{report.final_loss:.4f}", "—", "")
    console.print(table)
    if getattr(args, "report", None):
        write_markdown(report, Path(args.report))
        console.print(f"[dim]wrote report to {args.report}[/]")
    if report.passed():
        console.print("[bold green]GATE PASSED[/]")
        return 0
    console.print(f"[bold red]GATE FAILED[/]: {', '.join(report.failures())}")
    return 1


def _cmd_gradcheck(_args) -> int:
    x, y = make_spiral()
    x_train, y_train, _, _ = train_test_split(x, y)
    rel = gradient_check(MLP((2, 64, 3)), x_train[:50], y_train[:50])
    console.print(f"max relative error (analytical vs numerical): [bold]{rel:.2e}[/]")
    console.print("[green]backprop verified[/]" if rel <= 1e-4 else "[red]gradient mismatch[/]")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nnscratch", description="An MLP built from scratch in numpy.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_bench = sub.add_parser("benchmark", help="Train + gradient-check + gate.")
    p_bench.add_argument("--report", default=None, help="Write a markdown report to this path.")
    p_bench.set_defaults(func=_cmd_benchmark)

    sub.add_parser("gradcheck", help="Verify backprop against numerical gradients.").set_defaults(
        func=_cmd_gradcheck
    )
    sub.add_parser("demo", help="Train + gate.").set_defaults(func=_cmd_benchmark)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
