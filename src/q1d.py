"""Question 1d: Training loop, and evaluation"""

import numpy as np
import matplotlib.pyplot as plt
from .q1b import MLP, bce_loss

def train(model, X_train, y_train, X_val, y_val,
          lr=0.05, batch_size=32, epochs=200, random_state=42):
    """Mini-batch gradient descent.  Returns loss history dict."""
    rng = np.random.RandomState(random_state)
    N = X_train.shape[0]
    history = {"train_loss": [], "val_loss": []}

    for _ in range(epochs):
        perm = rng.permutation(N)
        X_shuf = X_train[perm]
        y_shuf = y_train[perm]

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            xb, yb = X_shuf[start:end], y_shuf[start:end]
            y_hat = model.forward(xb)
            model.backward(y_hat, yb)
            model.update(lr)

        history["train_loss"].append(bce_loss(model.forward(X_train), y_train))
        history["val_loss"].append(bce_loss(model.forward(X_val), y_val))

    return history

def plot_losses(history):
    """Training and validation loss vs. epoch."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["train_loss"], label="Training loss")
    ax.plot(history["val_loss"],   label="Validation loss")
    ax.set_xlabel("Epoch index")
    ax.set_ylabel("Binary Cross Entropy Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    plt.tight_layout()
    plt.show()
    return fig


def plot_bounds(model, X_val, y_val):
    """A visualisation of the learned decision boundary overlaid on the validation set scatter plot.
To do this, evaluate the network on a fine grid covering [−2, 3] × [−1.5, 2] and use a filled
contour plot at threshold 0.5."""
    h = 0.02
    x_range = np.arange(-2, 3 + h, h)
    y_range = np.arange(-1.5, 2 + h, h)
    xx, yy = np.meshgrid(x_range, y_range)
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.forward(grid).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.3, colors=["blue", "red"])
    ax.contour(xx, yy, Z, levels=[0.5], colors="k", linewidths=1.5)
    y_flat = y_val.ravel()
    ax.scatter(X_val[:, 0], X_val[:, 1],
               c=y_flat, cmap="bwr", edgecolors="k", alpha=0.8, s=40)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Decision Boundary (Validation Set)")
    ax.set_xlim(-2, 3)
    ax.set_ylim(-1.5, 2)
    plt.tight_layout()
    plt.show()
    return fig

def compute_accuracy(model, X, y):
    """Classification accuracy in percent."""
    preds = (model.forward(X) >= 0.5).astype(int)
    return float(np.mean(preds == y) * 100)



# main function
def main(dataset, layer_sizes=(2, 16, 1), model_seed=42,
         lr=0.05, batch_size=32, epochs=200, train_seed=42):
    """Main training"""
    model = MLP(layer_sizes=layer_sizes, random_state=model_seed)

    history = train(
        model,
        dataset.X_train, dataset.y_train,
        dataset.X_val, dataset.y_val,
        lr=lr, batch_size=batch_size, epochs=epochs, random_state=train_seed,
    )

    plot_losses(history)
    plot_bounds(model, dataset.X_val, dataset.y_val)

    acc = compute_accuracy(model, dataset.X_val, dataset.y_val)
    print(f"\nFinal validation accuracy: {acc:.2f}%")

    return {"model": model, "history": history, "accuracy": acc}