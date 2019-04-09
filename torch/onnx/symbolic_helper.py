import torch
from torch._C import ListType, OptionalType
from torch.nn.modules.utils import _single, _pair, _triple
import warnings

import torch.onnx
# This import monkey-patches graph manipulation methods on Graph, used for the
# ONNX symbolics
import torch.onnx.utils

from functools import partial, wraps

import numpy
import math

__all__ = [ '_parse_arg',
            'parse_args',
            '_maybe_get_const',
            '_maybe_get_scalar',
            '_get_const',
            '_unpack_list',
            '_scalar',
            '_if_scalar_type_as',
            '_is_value',
            '_is_tensor_list',
            '_unimplemented',
            '_try_get_scalar_type',
            '_default_onnx_opset_version',
            '_onnx_master_opset',
            '_onnx_stable_opsets',
            '_export_onnx_opset_version',
            'cast_pytorch_to_onnx',
            'scalar_name_to_pytorch',
            'scalar_type_to_pytorch_type', 
            '_cast_func_template',
            'scalar_type_to_onnx'
    ]

# EDITING THIS FILE? READ THIS FIRST!
#
# - This file is ONLY for ATen operators (e.g., operators that show up in the
#   trace as aten::blah).  If you need to special case a primitive operator,
#   look at _run_symbolic_function
# - Parameter ordering does NOT necessarily match what is in VariableType.cpp;
#   tensors are always first, then non-tensor arguments.
# - Parameter names must *exactly* match the names in VariableType.cpp, because
#   dispatch is done with keyword arguments.
# - Looking for inplace ops?  They're detected by the trailing underscore, and
#   transparently dispatched to their non inplace versions in
#   'run_symbolic_function'.   See Note [Export inplace]
#
# ----------------------------------------------------------------------------------
# A note on Tensor types
# ----------------------------------------------------------------------------------
#
# In general, we should avoid depending on the type of Tensor Values contained
# within the trace graph. However, this is sometimes unavoidable (due to ONNX
# spec requirements, etc). If you are implementing a symbolic and need Tensor
# type information, note that there are several levels of Tensor types, defined
# in aten/src/ATen/core/jit_type.h:
#
# TensorType - This is a Tensor, but we don't know anything about its
#               properties (e.g. scalar type, # dims, shapes).
#               Appears as `Tensor` in graph print-outs.
# DimensionedTensorType <: TensorType - Denotes a Tensor for which we know the scalar
#                             type and number of dimensions, but not the concrete
#                             shapes. For example, appears as 'Float(*, *)' in
#                             graph print-outs. Useful accessor methods include
#                             dim() and scalarType()
# CompleteTensorType <: DimensionedTensorType - Denotes a Tensor for which we know the
#                                               concrete sizes in addition to the information
#                                               contained in TensorTyper. This adds a sizes()
#                                               method which can be used to retrieve the
#                                               concrete sizes.
#
# In general, we should prefer to rely on the least specific information possible.
# For example, not relying on tensor properties at all is better than relying
# on the number of dimensions (DimensionedTensorType) which is better than relying on
# concrete shapes (CompleteTensorType). Doing so will make the export symbolics
# more robust to different graphs.


# ---------------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------------

# Save some builtins as locals, because we'll shadown them below
_sum = sum


def _parse_arg(value, desc):
    if desc == 'none':
        return value
    if desc == 'v' or not _is_value(value):
        return value
    if value.node().kind() != 'onnx::Constant':
        raise RuntimeError("ONNX symbolic expected a constant value in the trace")
    tval = value.node()['value']
    if desc == 'i':
        return int(tval)
    elif desc == 'f':
        return float(tval)
    elif desc == 't':
        return tval
    elif desc == 'is':
        return [int(v) for v in tval]
    else:
        raise RuntimeError("Casting constants to `{}` is not implemented".format(desc))


def _maybe_get_const(value, desc):
    if _is_value(value) and value.node().kind() == 'onnx::Constant':
        return _parse_arg(value, desc)
    return value


def _maybe_get_scalar(value):
    value_t = _maybe_get_const(value, 't')
    if isinstance(value_t, torch.Tensor) and value_t.shape == ():
        return value_t
    return value


def _get_const(value, desc, arg_name):
    if _is_value(value) and value.node().kind() != 'onnx::Constant':
        raise RuntimeError("ONNX symbolic expected a constant value of the {} argument".format(arg_name))
    return _parse_arg(value, desc)


def _unpack_list(list_value):
    list_node = list_value.node()
    assert list_node.kind() == "prim::ListConstruct"
    return list(list_node.inputs())


def parse_args(*arg_descriptors):
    def decorator(fn):
        def wrapper(g, *args):
            # some args may be optional, so the length may be smaller
            assert len(arg_descriptors) >= len(args)
            args = [_parse_arg(arg, arg_desc) for arg, arg_desc in zip(args, arg_descriptors)]
            return fn(g, *args)
        # In Python 2 functools.wraps chokes on partially applied functions, so we need this as a workaround
        try:
            wrapper = wraps(fn)(wrapper)
        except Exception:
            pass
        return wrapper
    return decorator


def _scalar(x):
    """Convert a scalar tensor into a Python value."""
    assert x.numel() == 1
    return x.item()


def _if_scalar_type_as(g, self, tensor):
    """
    Convert self into the same type of tensor, as necessary.

    We only support implicit casting for scalars, so we never
    actually need to insert an ONNX cast operator here; just
    fix up the scalar.
    """
    if isinstance(self, torch._C.Value):
        return self
    elif tensor.type().kind() == "DimensionedTensorType" or tensor.type().kind() == "CompleteTensorType":
        ty = tensor.type().scalarType().lower()
        return getattr(self, ty)()
    else:
        return self


def _is_value(x):
    return isinstance(x, torch._C.Value)


def _is_tensor_list(x):
    return x.type().isSubtypeOf(ListType.ofTensors())


def _unimplemented(op, msg):
    warnings.warn("ONNX export failed on " + op + " because " + msg + " not supported")


def _try_get_scalar_type(*args):
    for arg in args:
        try:
            return arg.type().scalarType()
        except RuntimeError:
            pass
    return None


# ---------------------------------------------------------------------
# ONNX operator version
# ---------------------------------------------------------------------

# READ ME BEFORE EDITING _default_onnx_opset_version:
#
# The variable below controls which ONNX operator set version we are
# targeting. THIS VARIABLE HAS SEMANTIC EFFECT! Say a breaking
# change occurred in version 8. As long as this variable < 8, you can
# export models targeting the old behavior. However, if you bump
# this variable to 8 or later, the breaking change will take into effect:
# you MUST adjust any symbolic affected by breaking changes. The ONNX
# spec publishes a *comprehensive* list of BC-breaking changes for every
# operator revision at:
#
#   https://github.com/onnx/onnx/blob/master/docs/Changelog.md
#
# Please be sure to go through and check all of our implementations here before
# increasing this number. This includes symbolic definitions NOT in this
# file, so grep for "OpName" (with quotes)
#
# Besides, opset_version can be specified in the invocation of export()
# and export_to_pretty_string(), and _export_onnx_opset_version will be set
# and the symbolic functions should check it to determine the behavior
# of the exporter.


_default_onnx_opset_version = 9
_onnx_master_opset = 10
_onnx_stable_opsets = [9]
_export_onnx_opset_version = _default_onnx_opset_version


def _set_opset_version(opset_version):
    global _export_onnx_opset_version
    if opset_version == _default_onnx_opset_version:
        return
    if opset_version in _onnx_stable_opsets + [_onnx_master_opset]:
        _export_onnx_opset_version = opset_version
        return
    raise ValueError("Unsupported ONNX opset version: " + str(opset_version))

"""

# add new opsets at the begining of the map to keep it sorted
# from the most recent to the oldest opset number, and keep
# prioritizing the newest op versions over the oldest ones
_symbolic_opset_files = {
    10 :torch.onnx.symbolic_opset10,
    9 : torch.onnx.symbolic_opset9
}

def _is_exportable_aten_op():    
    for opset, symbolic_file in _symbolic_opset_files :
        if opset >= _export_onnx_opset_version and hasattr(symbolic_file, op_name):
            return True, getattr(symbolic_file, op_name)
    return False, None
"""

# Metaprogram symbolics for each ATen native specialized cast operator.
# For e.g. we specify a function named `_cast_uint8_t` that instantiates an
# ONNX cast node with `to` attribute 'UINT8'
#
# TODO: remove these once we support Type's in the JIT IR and we can once again
# use the unified toType operator
cast_pytorch_to_onnx = {
    'Byte': torch.onnx.TensorProtoDataType.UINT8,
    'Char': torch.onnx.TensorProtoDataType.INT8,
    'Double': torch.onnx.TensorProtoDataType.DOUBLE,
    'Float': torch.onnx.TensorProtoDataType.FLOAT,
    'Half': torch.onnx.TensorProtoDataType.FLOAT16,
    'Int': torch.onnx.TensorProtoDataType.INT32,
    'Long': torch.onnx.TensorProtoDataType.INT64,
    'Short': torch.onnx.TensorProtoDataType.INT16,
}

scalar_name_to_pytorch = {
    'uint8_t': 'Byte',
    'int8_t': 'Char',
    'double': 'Double',
    'float': 'Float',
    'half': 'Half',
    'int': 'Int',
    'int64_t': 'Long',
    'int16_t': 'Short',
}


# This indicates each scalar type's corresponding
# torch type. Related source:
# https://github.com/pytorch/pytorch/blob/da7468853ae322252270bbb58032668bd21b7457/c10/core/ScalarType.h
scalar_type_to_pytorch_type = [
    torch.uint8,    # 0
    torch.int8,     # 1
    torch.short,    # 2
    torch.int,      # 3
    torch.int64,    # 4
    torch.half,     # 5
    torch.float,    # 6
    torch.double,   # 7
]


def _cast_func_template(to_i, g, input, non_blocking):
    return g.op("Cast", input, to_i=to_i)


for k, v in cast_pytorch_to_onnx.items():
    name = '_cast_{}'.format(k)
    globals()[name] = parse_args('v', 'i')(partial(_cast_func_template, v))


scalar_type_to_onnx = [
    cast_pytorch_to_onnx["Byte"],
    cast_pytorch_to_onnx["Char"],
    cast_pytorch_to_onnx["Short"],
    cast_pytorch_to_onnx["Int"],
    cast_pytorch_to_onnx["Long"],
    cast_pytorch_to_onnx["Half"],
    cast_pytorch_to_onnx["Float"],
    cast_pytorch_to_onnx["Double"],
]

class SymbolicRegistry:
    _domain = None
    _version = None
    _registry = None

    import torch.onnx.symbolic_opset9
    import torch.onnx.symbolic_opset10
    _symbolic_versions = {
        9 : torch.onnx.symbolic_opset9,
        10 : torch.onnx.symbolic_opset10
    }

    def __init__(self, domain, version):
        self._registry = {}
        self.register_version(domain, version)

    def register_version(self, domain, version):
        self._set_version(domain, version)
        if not self.is_registered_version(domain, version):
            self._register_version()

    def _register_version(self):
        self._registry[self._domain] = {}
        self._registry[self._domain][self._version] = {}
        
        it_version = self._version
        while it_version >= 9:
            symbolic_version = self._symbolic_versions[it_version].__dict__
            for opname in symbolic_version:
                if not self.is_registered_op(opname):
                    self.register_op(opname, symbolic_version[opname])
            it_version = it_version - 1

    def _set_version(self, domain, version):
        if self._symbolic_versions[version] is None: # or domain not supported
            warnings.warn("ONNX export failed. Opset version {} is not supported".format(version))
        self._version = version
        self._domain = domain

    def is_registered_version(self, domain, version):
        return domain in self._registry and version in self._registry[domain] 

    def register_op(self, opname, op, domain=None, version=None):
        if domain is None:
            domain = self._domain
        if version is None:
            version = self._version
        self._registry[domain][version][opname] = op

    def is_registered_op(self, opname, domain=None, version=None):
        if domain is None:
            domain = self._domain
        if version is None:
            version = self._version
        return domain in self._registry and version in self._registry[domain] and opname in self._registry[domain][version]

    def get_registered_op(self, opname, domain=None, version=None):
        if domain is None:
            domain = self._domain
        if version is None:
            version = self._version
        return self._registry[domain][version][opname]
