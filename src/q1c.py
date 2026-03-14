"""Question 1c: Backward-pass"""

import numpy as np
from .q1b import MLP, bce_loss # import the loss and the model from q1b


def gradient_check(model, X, y, epsilon=1e-5):
    """Compare analytical gradients with finite-difference approximation
    on a small fixed subset of parameters.
    """
    y_hat = model.forward(X)
    model.backward(y_hat, y)

    max_diff = 0.0
    records = []

    checks = [
        ("linear1", "W", model.linear1.W, model.linear1.dW, [(0, 0), (1, 5), (0, 10)]),
        ("linear1", "b", model.linear1.b, model.linear1.db, [(0,), (5,), (10,)]),
        ("linear2", "W", model.linear2.W, model.linear2.dW, [(0, 0), (5, 0), (10, 0)]),
        ("linear2", "b", model.linear2.b, model.linear2.db, [(0,)]),
    ]

    for layer_name, param_name, param, grad, indices in checks:
        for idx in indices:
            original = param[idx]

            param[idx] = original + epsilon
            loss_plus = bce_loss(model.forward(X), y)

            param[idx] = original - epsilon
            loss_minus = bce_loss(model.forward(X), y)

            param[idx] = original

            numerical = (loss_plus - loss_minus) / (2 * epsilon)
            analytical = grad[idx]
            diff = abs(numerical - analytical)

            max_diff = max(max_diff, diff)
            records.append((f"{layer_name}.{param_name}{list(idx)}", analytical, numerical, diff))

    return max_diff, records


def print_gradient_shapes(model):
    """Print gradient tensor shapes as a check"""
    shapes = [
        ("dL/dW2", model.linear2.dW.shape, "(16, 1)"),
        ("dL/db2", model.linear2.db.shape, "(1,)"),
        ("dL/dW1", model.linear1.dW.shape, "(2, 16)"),
        ("dL/db1", model.linear1.db.shape, "(16,)"),
    ]
    print("Gradient shapes:")

    for k, l, m in shapes:
        ok = "OK" if str(l) == m else "MISMATCH"
        print(f"{k:10s}  {str(l):10s}  expected {m:10s}  {ok}")


def main(dataset, layer_sizes=(2, 16, 1), model_seed=42, epsilon=1e-5):
    """Run the gradient check, print results, return max absolute difference."""
    model = MLP(layer_sizes=layer_sizes, random_state=model_seed)
    max_diff, records = gradient_check(
        model, dataset.X_train, dataset.y_train, epsilon=epsilon,
    )

    print_gradient_shapes(model)
    n_params = len(records)
    print(f"\nGradient check (eps = {epsilon}, {n_params} parameters checked)")
    print(f"Max |analytical - numerical|: {max_diff:.2e}")

    print("\nChecked parameters:")
    for name, analytical, numerical, diff in records:
        print(f"{name:20s} analytical={analytical: .6e} numerical={numerical: .6e} diff={diff: .2e}")

    return max_diff