
# import tensorflow as tf
import torch as t   # type: ignore

# alpha = 1.6732632423543772848170429916717
alpha: float = 1.67
beta: float = 1.0507009873554804934193349852946

__doc__ = """
This module contains the B-SiLU activation function and its derivative.
"""

def BSiLU_CONFIG(alpha: float = 1.67) -> float:
    _alpha: float = alpha
    return _alpha

def sigma(x: t.Tensor) -> t.Tensor:
    """
    Sigmoid avtivation function.
    """
    S: t.Tensor = 1 / (1 + t.exp(-x))
    return S

def BSiLU(x: t.Tensor) -> t.Tensor:
    """
    B-SiLU math function.
    """
    BSiLU: t.Tensor = (x + alpha) * sigma(x) - alpha / 2
    return BSiLU

def BSiLU_derivative(x: t.Tensor) -> t.Tensor:
    """
    B-SiLU derivative math function.
    """
    BSiLU_derivative: t.Tensor = sigma(x) + (x + alpha) * sigma(x) * (1 - sigma(x))
    return BSiLU_derivative