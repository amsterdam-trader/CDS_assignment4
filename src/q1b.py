f"""Question 1b: MLP architecture, forward pass, and loss function.

Architecture
------------
Input(2)  -->  Linear(2,16) -> Sigmoid  -->  Linear(16,1) -> Sigmoid  -->  prediction
"""

import numpy as np

# helpers
def _sigmoid(z):
    """sigmoid function"""
    return np.where(
        z >= 0,
        1 / (1 + np.exp(-z)),
        np.exp(z) / (1 + np.exp(z)),
    )


def bce_loss(y_hat, y):
    """Binary cross-entropy (mean over batch)."""
    eps = 1e-15
    yh = np.clip(y_hat, eps, 1 - eps)
    return -np.mean(y * np.log(yh) + (1 - y) * np.log(1 - yh))

# Layers
class LinearLayer:
    """Fully-connected layer z = xW + b"""

    def __init__(self, n_in, n_out):
        limit = np.sqrt(6 / (n_in + n_out))
        self.W = np.random.uniform(-limit, limit, size=(n_in, n_out))
        self.b = np.zeros(n_out)
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dz):
        """Compute and store parameter gradients
        """
        self.dW = self.x.T @ dz
        self.db = np.sum(dz, axis=0)
        return dz @ self.W.T

    def update(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db


class SigmoidLayer:
    """sigmoid activation."""

    def __init__(self):
        self.z = None

    def forward(self, z):
        self.z = z
        return _sigmoid(z)

    def backward(self, da):
        s = _sigmoid(self.z)
        return da * s * (1 - s)

# The main model:
class MLP:
    """multi layer perceptron """
    def __init__(self, layer_sizes, random_state):
        state = np.random.get_state()
        np.random.seed(random_state)
        self.linear1 = LinearLayer(layer_sizes[0], layer_sizes[1])
        self.sigmoid1 = SigmoidLayer()
        self.linear2 = LinearLayer(layer_sizes[1], layer_sizes[2])
        self.sigmoid2 = SigmoidLayer()
        np.random.set_state(state)

    def forward(self, x):
        z1 = self.linear1.forward(x)
        a1 = self.sigmoid1.forward(z1)
        z2 = self.linear2.forward(a1)
        return self.sigmoid2.forward(z2)

    def backward(self, y_hat, y):
        """Back-propagate through the network (does **not** update weights).

        The combined sigmoid-output + BCE gradient simplifies to prediction - actual.
        We divide by N here so all downstream gradients are w.r.t. the
        *mean* loss.
        """
        N = y_hat.shape[0]
        dz = (y_hat - y) / N            # Step 1: output gradient
        da = self.linear2.backward(dz)   # Step 2: linear-2
        dz = self.sigmoid1.backward(da)  # Step 3: sigmoid-1
        self.linear1.backward(dz)        # Step 4: linear-1

    def update(self, lr):
        """Gradient-descent logic"""
        self.linear1.update(lr)
        self.linear2.update(lr)


def main(dataset, layer_sizes, model_seed):
    """Initialize the model, run one forward pass on training data, return y_hat.
    """
    model = MLP(layer_sizes=layer_sizes, random_state=model_seed)
    y_hat = model.forward(dataset.X_train)
    loss = bce_loss(y_hat, dataset.y_train)
    print(f"Initial Cross entropy loss: {loss:.4f}")
    return y_hat