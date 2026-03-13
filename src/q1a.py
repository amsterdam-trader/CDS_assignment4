"""Question 1a"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class Dataset:
    """Variable declaration
    """
    X_train: np.ndarray
    X_val: np.ndarray
    y_train: np.ndarray  # shape (N_train, 1)
    y_val: np.ndarray    # shape (N_val, 1)


def generate_moons(n_samples=300, noise=0.2, random_state=80):
    """Generate the two-moons binary classification dataset."""
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    return X, y.reshape(-1, 1)


def plot_scatter(X, y, title="Two-Moons Dataset"):
    """Scatter plot of the dataset coloured by class label."""
    fig, ax = plt.subplots(figsize=(7, 5))
    y_flat = y.ravel()
    scatter = ax.scatter(
        X[:, 0], X[:, 1],
        c=y_flat, cmap="bwr", edgecolors="k", alpha=0.7, s=30,
    )
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(title)
    fig.colorbar(scatter, ax=ax, label="Class")
    plt.tight_layout()
    plt.show()
    return fig


def split_data(X, y, test_size=0.2, random_state=42):
    """80/20 stratified train/validation split."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )


def main(n_samples=300, noise=0.2, data_seed=80, split_seed=42, test_size=0.2):
    """Generate the moons dataset, show scatter plot, return Dataset."""
    X, y = generate_moons(n_samples=n_samples, noise=noise, random_state=data_seed)
    plot_scatter(X, y)
    X_train, X_val, y_train, y_val = split_data(
        X, y, test_size=test_size, random_state=split_seed,
    )
    dataset = Dataset(X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val)
    print(f"Training set:   {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    return dataset