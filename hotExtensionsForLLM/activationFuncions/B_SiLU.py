
# import tensorflow as tf
import torch as t

# alpha = 1.6732632423543772848170429916717
alpha: float = 1.67
beta: float = 1.0507009873554804934193349852946

__doc__ = """
This module contains the B-SiLU activation function and its derivative.
"""

def BSiLU_CONFIG(alpha=1.67):
    alpha = alpha
    return alpha

def sigma(x):
    """
    Sigmoid avtivation function.
    """
    S = 1 / (1 + t.exp(-x))
    return S

def BSiLU(x):
    """
    B-SiLU math function.
    """
    BSiLU = (x + alpha) * sigma(x) - alpha / 2
    return BSiLU

def BSiLU_derivative(x):
    """
    B-SiLU derivative math function.
    """
    BSiLU_derivative = sigma(x) + (x + alpha) * sigma(x) * (1 - sigma(x))
    return BSiLU_derivative