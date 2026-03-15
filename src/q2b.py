import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------
# Derivative of Sigmoid Function
# --------------------------------------------------------------------------------------

def sigmoid(z):
    return np.exp(-z) / (1 + np.exp(-z))**2

# --------------------------------------------------------------------------------------
# Derivative of ReLU Function
# --------------------------------------------------------------------------------------

def relu(z):
    return np.where(z > 0, 1, 0)

def main():
    # define z = [-4, 4]
    z = np.linspace(-4, 5, 200) 

    fig, ax = plt.subplots(1, 2, figsize = (10, 4))

    ax[0].plot(z, sigmoid(z))
    ax[0].set_title("Derivative Sigmoid Function")
    ax[0].set_xlabel("z")
    ax[0].set_ylabel("σ'(z)")
    
    ax[1].plot(z, relu(z))
    ax[1].set_title("Derivative ReLU Function")
    ax[1].set_xlabel("z")
    ax[1].set_ylabel("ReLU'(z)")
    plt.show()
    