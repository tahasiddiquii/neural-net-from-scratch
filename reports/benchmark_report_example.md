# neural-net-from-scratch — report

An MLP (2 → 64 → 3, ReLU) trained on a 3-arm spiral with pure-numpy backprop and SGD.

| metric | value | threshold |
| --- | --- | --- |
| train_accuracy | 0.9931 | — |
| test_accuracy | 0.9722 | ≥ 0.90 |
| gradient_check (max rel err) | 3.52e-08 | ≤ 1e-04 |
| final_loss | 0.0176 | — |

**Gate: PASSED**

The gradient check is the important line: the analytical gradients from `backward` match central finite differences to ~1e-6, which *proves* the backprop is correct — the test accuracy then proves the network actually learns the non-linear boundary a linear model can't.
