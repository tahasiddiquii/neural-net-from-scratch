# Agent / contributor guide

Orientation for an AI agent or new contributor working in this repo.

## What this is

A multilayer perceptron built **from scratch in pure numpy** — no PyTorch, no TensorFlow, no
sklearn. Forward pass, hand-derived backprop, SGD with momentum, trained on a non-linear
spiral. A **gradient check** proves the math and an accuracy gate proves it learns.

## Layout

```
src/nnscratch/
  data.py       spiral generator + train/test split
  layers.py     Linear + ReLU (each owns forward/backward + its grads)
  losses.py     softmax + cross-entropy (with the combined gradient)
  network.py    MLP — stacks layers, exposes params_and_grads
  optim.py      SGD with momentum
  train.py      mini-batch training loop + accuracy
  gradcheck.py  numerical vs analytical gradient check
  benchmark.py  train + gradient-check + gate (THRESHOLDS)
  cli.py        benchmark / gradcheck / demo
tests/          one file per module
reports/        benchmark_report_example.md (committed proof)
```

## Conventions

- Python 3.11+ (CI pins 3.12). `from __future__ import annotations` everywhere. Ruff for
  lint/format (`make fmt`, `make lint`); line length 110.
- **No frameworks.** numpy only (rich is just for the CLI). The point is the mechanics.
- **Gradient-check anything you differentiate.** If you add a layer or loss, add a
  `gradient_check` assertion — it's the fastest way to catch a wrong derivative.
- **ReLU + finite differences hate kinks.** Gradient-check on a class-mixed batch; an
  all-one-class batch makes units uniformly dead and the numerical check unreliable.
- **Never hand-write metrics.** Every number comes from `run_benchmark()`. Change behavior,
  regenerate (`make report`), update the README.

## Definition of done

```bash
make lint       # ruff clean
make test       # all tests pass (incl. the gradient check)
make benchmark  # gate exits 0 (test_accuracy >= 0.90 and gradient check <= 1e-4)
```

The same checks run in CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)).

## Extending

- New activation (tanh, sigmoid): implement forward/backward + a gradient-check test.
- New layer (dropout, batchnorm): same interface (`params`/`grads`), add to the MLP.
- New optimizer (Adam): implement `step(net)` and swap it into `train`.
