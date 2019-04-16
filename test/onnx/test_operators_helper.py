from test_pytorch_common import TestCase, flatten

import torch
import torch.onnx
from torch.nn import Module
import torch.nn as nn

import itertools
import io
import inspect
import os


_onnx_test = False  # flag to produce onnx test cases.
_onnx_dep = True  # flag to import onnx package.


def export_to_pbtxt(model, inputs, *args, **kwargs):
    return torch.onnx.export_to_pretty_string(
        model, inputs, None, verbose=False, google_printer=True,
        *args, **kwargs)


def export_to_pb(model, inputs, *args, **kwargs):
    kwargs['operator_export_type'] = torch.onnx.OperatorExportTypes.ONNX
    f = io.BytesIO()
    with torch.no_grad():
        torch.onnx.export(model, inputs, f, *args, **kwargs)
    return f.getvalue()


class FuncModule(Module):
    def __init__(self, f, params=None):
        if params is None:
            params = ()
        super(FuncModule, self).__init__()
        self.f = f
        self.params = nn.ParameterList(list(params))

    def forward(self, *args):
        return self.f(*itertools.chain(args, self.params))


class TestOperators(TestCase):

    def assertONNX(self, f, args, params=None, **kwargs):
        if params is None:
            params = ()
        if isinstance(f, nn.Module):
            m = f
        else:
            m = FuncModule(f, params)
        m.eval()
        onnx_model_pbtxt = export_to_pbtxt(m, args, **kwargs)
        subname = kwargs.pop('subname', None)
        self.assertExpected(onnx_model_pbtxt, subname)
        if _onnx_dep:
            onnx_model_pb = export_to_pb(m, args, **kwargs)
            import onnx
            import onnx.checker
            import onnx.numpy_helper
            import test_onnx_common
            model_def = onnx.ModelProto.FromString(onnx_model_pb)
            onnx.checker.check_model(model_def)
            if _onnx_test:
                test_function = inspect.stack()[1][0].f_code.co_name
                test_name = test_function[0:4] + "_operator" + test_function[4:]
                output_dir = os.path.join(test_onnx_common.pytorch_operator_dir, test_name)
                # Assume:
                #     1) the old test should be delete before the test.
                #     2) only one assertONNX in each test, otherwise will override the data.
                assert not os.path.exists(output_dir), "{} should not exist!".format(output_dir)
                os.makedirs(output_dir)
                with open(os.path.join(output_dir, "model.onnx"), 'wb') as file:
                    file.write(model_def.SerializeToString())
                data_dir = os.path.join(output_dir, "test_data_set_0")
                os.makedirs(data_dir)
                if isinstance(args, Variable):
                    args = (args,)
                for index, var in enumerate(flatten(args)):
                    tensor = onnx.numpy_helper.from_array(var.data.numpy())
                    with open(os.path.join(data_dir, "input_{}.pb".format(index)), 'wb') as file:
                        file.write(tensor.SerializeToString())
                outputs = m(*args)
                if isinstance(outputs, Variable):
                    outputs = (outputs,)
                for index, var in enumerate(flatten(outputs)):
                    tensor = onnx.numpy_helper.from_array(var.data.numpy())
                    with open(os.path.join(data_dir, "output_{}.pb".format(index)), 'wb') as file:
                        file.write(tensor.SerializeToString())

    def assertONNXRaises(self, err, f, args, params=None, **kwargs):
        if params is None:
            params = ()
        if isinstance(f, nn.Module):
            m = f
        else:
            m = FuncModule(f, params)
        self.assertExpectedRaises(err, lambda: export_to_pbtxt(m, args, **kwargs))

    def assertONNXRaisesRegex(self, err, reg, f, args, params=None, **kwargs):
        if params is None:
            params = ()
        if isinstance(f, nn.Module):
            m = f
        else:
            m = FuncModule(f, params)
        with self.assertRaisesRegex(err, reg):
            export_to_pbtxt(m, args, **kwargs)
