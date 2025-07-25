from .hotExt4LLM import (
    BSiLU, BSiLU_derivative,
    NeLU, NeLU_derivative,
    PnPNystromAttention
)

__all__ = [
    'BSiLU', 'BSiLU_derivative',
    'NeLU', 'NeLU_derivative',

    'PnPNystromAttention'
    ]


__package__ = 'hotExt4LLM'
__author__ = 'Tourist Chen'
__VERSION__ = '0.1.0'

__doc__ = """
'hotExt4LLM' is a package for hot extensions for LLMS.

These are 'PnP' modules, include 'Activation Functions' and 'Attention Functions' and so on.
'PnP' means 'Plug-and-Play', which means you can easily use these modules in your own projects.

This package is under development, and more functions will be added in the future.
"""

