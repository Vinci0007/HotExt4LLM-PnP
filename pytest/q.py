import os
import sys
import torch as _t

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)


from hotExt4LLM import BSiLU


# import hotExtensionsForLLM.activationFunctions.B_SiLU

test_tensor: _t.Tensor = _t.rand(2, 100, 64)

output = BSiLU(test_tensor)

print(f"Output shape: {output.shape}")

# from __future__ import annotations


