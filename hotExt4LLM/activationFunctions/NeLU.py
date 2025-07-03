
import torch as t   

__doc__ = """
This is the implementation of the NeLU activation function.

Bounded Sigmoid Linear Unit (NeLU) 
- A new activation function that combines the benefits of ReLU and Leaky ReLU. 
- It is a smooth and non-monotonic function that is bounded between -1 and 1. 
- It can be used in Transformers and other deep learning models. 
"""

alpha: float = 0.2

def NeLU(x: t.Tensor) -> t.Tensor:
    """
    This is the implementation of the NeLU activation math function.
    """
    return x * (x > 0) + (-alpha / (1 + x ** 2)) * (x < 0)


def NeLU_derivative(x: t.Tensor) -> t.Tensor:
    """
    This is the implementation of the NeLU activation derivative function.
    """
    return 1 * (x > 0) + (alpha * (x * 2 / ((1 + x ** 2) ** 2))) * (x < 0)