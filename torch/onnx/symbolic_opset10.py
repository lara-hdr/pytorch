import torch
from torch._C import ListType, OptionalType
from torch.nn.modules.utils import _single, _pair, _triple
import warnings

import torch.onnx
# This import monkey-patches graph manipulation methods on Graph, used for the
# ONNX symbolics
import torch.onnx.utils
import torch.onnx.symbolic_helper
import torch.onnx.symbolic_opset9

from functools import partial, wraps

# Add new operator here

@parse_args('v', 'i', 'i', 'i', 'i')
def topk(g, self, k, dim, largest, sorted, out=None):
    _unimplemented("TopK", "Testing dispatch")