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

from torch.onnx.symbolic_helper import *
from torch.onnx.symbolic_opset9 import *

from functools import partial, wraps

# Add new operator here
@parse_args('v', 'i', 'i', 'i', 'i')
def topk(g, self, k, dim, largest, sorted, out=None):
    if out is not None:
        _unimplemented("TopK", "Out parameter is not supported for topk")
    if not largest:
        _unimplemented("TopK", "Ascending TopK is not supported")
    k = g.op("Constant", value_t=torch.tensor(k, dtype=torch.int64))
    k = unsqueeze(g, k, 0)
    return g.op("TopK", self, k, axis_i=dim, outputs=2)
