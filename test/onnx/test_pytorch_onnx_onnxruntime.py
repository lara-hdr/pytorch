from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import onnxruntime  # noqa
import torch

import numpy as np
import io

from test_pytorch_common import skipIfUnsupportedMinOpsetVersion, skipIfUnsupportedOpsetVersion
from test_pytorch_common import BATCH_SIZE, RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, RNN_SEQUENCE_LENGTH

import model_defs.word_language_model as word_language_model


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def validate_ORT_prediction(ort_sess, input, output, rtol=1e-05, atol=1e-08):
    input, _ = torch.jit._flatten(input)
    output, _ = torch.jit._flatten(output)

    inputs = list(map(to_numpy, input))
    outputs = list(map(to_numpy, output))

    # compute onnxruntime output prediction
    ort_inputs = dict((ort_sess.get_inputs()[i].name, input) for i, input in enumerate(inputs))
    ort_outs = ort_sess.run(None, ort_inputs)

    # compare onnxruntime and PyTorch results
    assert len(outputs) == len(ort_outs), "number of outputs differ"

    # compare onnxruntime and PyTorch results
    [np.testing.assert_allclose(out, ort_out, rtol=rtol, atol=atol) for out, ort_out in zip(outputs, ort_outs)]

def run_model_test(self, model, train, batch_size=2, state_dict=None,
                   input=None, use_gpu=True, rtol=0.001, atol=1e-7,
                   example_outputs=None, do_constant_folding=True,
                   dynamic_axes=None, fixed_batch_size=False,
                   validate_prediction=True):
    if not train:
        model.eval()

    if input is None:
        input = torch.randn(batch_size, 3, 224, 224, requires_grad=True)

    with torch.no_grad():
        if isinstance(input, torch.Tensor):
            input = (input,)
        output = model(*input)
        if isinstance(output, torch.Tensor):
            output = (output,)

        # export the model to ONNX
        f = io.BytesIO()
        torch.onnx._export(model, input, f,
                           opset_version=self.opset_version,
                           example_outputs=output,
                           dynamic_axes=dynamic_axes,
                           do_constant_folding=do_constant_folding,
                           fixed_batch_size=fixed_batch_size)

        ort_sess = onnxruntime.InferenceSession(f.getvalue())
        if validate_prediction:
            validate_ORT_prediction(ort_sess, input, output, rtol=rtol, atol=atol)
        return ort_sess


class TestONNXRuntime(unittest.TestCase):
    from torch.onnx.symbolic_helper import _export_onnx_opset_version
    opset_version = _export_onnx_opset_version

    def run_test(self, model, input, rtol=1e-3, atol=1e-7,
                 do_constant_folding=False, dynamic_axes=None,
                 fixed_batch_size=False, validate_prediction=True):
        return run_model_test(self, model, False, None,
                              input=input, rtol=rtol, atol=atol,
                              do_constant_folding=do_constant_folding,
                              dynamic_axes=dynamic_axes,
                              fixed_batch_size=fixed_batch_size,
                              validate_prediction=True)

    def run_word_language_model(self, model_name):
        ntokens = 50
        emsize = 5
        nhid = 5
        nlayers = 5
        dropout = 0.2
        tied = False
        batchsize = 5
        model = word_language_model.RNNModel(model_name, ntokens, emsize,
                                             nhid, nlayers, dropout, tied,
                                             batchsize)
        x = torch.arange(0, ntokens).long().view(-1, batchsize)
        # Only support CPU version, since tracer is not working in GPU RNN.
        self.run_test(model, (x, model.hidden))

    def test_word_language_model_RNN_TANH(self):
        self.run_word_language_model("RNN_TANH")

    def test_word_language_model_RNN_RELU(self):
        self.run_word_language_model("RNN_RELU")

    def test_word_language_model_LSTM(self):
        self.run_word_language_model("LSTM")

    def test_word_language_model_GRU(self):
        self.run_word_language_model("GRU")

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_full_trace(self):
        class FullModel(torch.nn.Module):
            def forward(self, x):
                return torch.full((3, 4), x, dtype=torch.long)

        x = torch.tensor(12)
        self.run_test(FullModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_full_script(self):
        class FullModelScripting(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                return torch.full((3, 4), x, dtype=torch.long)

        x = torch.tensor(12)
        self.run_test(FullModelScripting(), x)

    def test_maxpool(self):
        model = torch.nn.MaxPool1d(2, stride=1)
        x = torch.randn(20, 16, 50)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(8)
    def test_maxpool_with_indices(self):
        model = torch.nn.MaxPool1d(2, stride=1, return_indices=True)
        x = torch.randn(20, 16, 50)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_maxpool_dilation(self):
        model = torch.nn.MaxPool1d(2, stride=1, dilation=2)
        x = torch.randn(20, 16, 50)
        self.run_test(model, x)

    def test_avgpool(self):
        model = torch.nn.AvgPool1d(2, stride=1)
        x = torch.randn(20, 16, 50)
        self.run_test(model, x)

    def test_slice_trace(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                return x[0:1]

        x = torch.randn(3)
        self.run_test(MyModule(), x)

    def test_slice_script(self):
        class DynamicSliceModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                return x[1:x.size(0)] 

        x = torch.rand(1, 2)
        self.run_test(DynamicSliceModel(), x)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_flip(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                return torch.flip(x, dims=[0])

        x = torch.tensor(np.arange(6.0).reshape(2, 3))
        self.run_test(MyModule(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_interpolate_scale(self):
        class MyModel(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.interpolate(x, mode="nearest", scale_factor=2)
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.run_test(MyModel(), x)

    # NOTE: Supported in onnxruntime master, enable this after 0.5 release.
    @skipIfUnsupportedOpsetVersion([10])
    def test_interpolate_output_size(self):
        class MyModel(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.interpolate(x, mode="nearest", size=(6, 8))
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.run_test(MyModel(), x)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_interpolate_downsample(self):
        class MyModel(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.interpolate(x, mode="nearest", scale_factor=[1, 1, 0.5, 0.5])
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.run_test(MyModel(), x)

    # TODO: enable for opset 10 when ONNXRuntime version will be updated 
    @skipIfUnsupportedOpsetVersion([10])
    def test_topk(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                return torch.topk(x, 3)

        x = torch.arange(1., 6., requires_grad=True)
        self.run_test(MyModule(), x)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_topk_script(self):
        class MyModuleDynamic(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x, k):
                return torch.topk(x, k)

        x = torch.arange(1., 6., requires_grad=True)
        k = torch.tensor(3)
        self.run_test(MyModuleDynamic(), [x, k])

    def test_layer_norm(self):
        model = torch.nn.LayerNorm([10, 10])
        x = torch.randn(20, 5, 10, 10)
        self.run_test(model, x, rtol=1e-05, atol=1e-06)

    def test_reduce_log_sum_exp(self):
        class ReduceLogSumExpModel(torch.nn.Module):
            def forward(self, input):
                a = torch.logsumexp(input, dim=0)
                b = torch.logsumexp(input, dim=(0, 1))
                return a + b

        x = torch.randn(4, 4, requires_grad=True)
        self.run_test(ReduceLogSumExpModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_lstm(self):
        model = torch.nn.LSTM(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, 1, bidirectional=False)
        model.eval()
        input = torch.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
        h0 = torch.randn(1, BATCH_SIZE, RNN_HIDDEN_SIZE)
        c0 = torch.randn(1, BATCH_SIZE, RNN_HIDDEN_SIZE)
        ort_session = self.run_test(model, (input, (h0, c0)), validate_prediction=False)

        # LSTM's output is (output, (h, c)) in PyTorch and (output, h, c) in ONNX
        # This is the reason we don't validate the ORT prediction in run_test for
        # all the lstm tests below. Instead we do it here.
        output, (h, c) = model(input, (h0, c0))
        validate_ORT_prediction(ort_session, (input, h0, c0), (output, h, c))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_lstm_default_init_state(self):
        model = torch.nn.LSTM(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, 1, bidirectional=False)
        model.eval()
        input = torch.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
        ort_session = self.run_test(model, input, validate_prediction=False)

        output, (h, c) = model(input)
        validate_ORT_prediction(ort_session, input, (output, h, c))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_lstm_fixed_batch_size(self):
        class LSTMModel(torch.nn.Module):
            def __init__(self):
                super(LSTMModel, self).__init__()
                self.lstm = torch.nn.LSTM(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, 1, bidirectional=False)

            def forward(self, input):
                batch_size = input.size()[1]
                h0_np = np.ones([1, batch_size, RNN_HIDDEN_SIZE]).astype(np.float32)
                c0_np = np.ones([1, batch_size, RNN_HIDDEN_SIZE]).astype(np.float32)
                h0 = torch.from_numpy(h0_np)
                c0 = torch.from_numpy(c0_np)
                return self.lstm(input, (h0, c0))

        model = LSTMModel()
        model.eval()

        input = torch.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
        ort_session = self.run_test(model, input, fixed_batch_size=True, validate_prediction=False)

        output, (h, c) = model(input)
        validate_ORT_prediction(ort_session, input, (output, h, c))

        # verify with different input of same batch size
        input2 = torch.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
        output, (h, c) = model(input2)
        validate_ORT_prediction(ort_session, input2, (output, h, c))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_lstm_post_fix_init_state(self):
        class LSTMModel(torch.nn.Module):
            def __init__(self):
                super(LSTMModel, self).__init__()
                self.lstm = torch.nn.LSTM(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE,
                                          1, bidirectional=False)

            def forward(self, input):
                batch_size = input.size()[1]
                h0_np = np.ones([1, batch_size, RNN_HIDDEN_SIZE]).astype(np.float32)
                c0_np = np.ones([1, batch_size, RNN_HIDDEN_SIZE]).astype(np.float32)
                h0 = torch.from_numpy(h0_np)
                c0 = torch.from_numpy(c0_np)
                return self.lstm(input, (h0, c0))

        model = LSTMModel()
        model.eval()

        input = torch.randn(RNN_SEQUENCE_LENGTH, 1, RNN_INPUT_SIZE)
        ort_session = self.run_test(model, input, dynamic_axes={'input' : {0 : 'seq', 1 : 'batch'}},
                                    validate_prediction=False)

        output, (h, c) = model(input)
        validate_ORT_prediction(ort_session, input, (output, h, c))

        # verify with different input of different batch size
        input2 = torch.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
        output, (h, c) = model(input2)
        validate_ORT_prediction(ort_session, input2, (output, h, c))

    @skipIfUnsupportedMinOpsetVersion(8)
    def test_adaptive_max_pool(self):
        model = torch.nn.AdaptiveMaxPool1d((5), return_indices=False)
        x = torch.randn(20, 16, 50, requires_grad=True)
        self.run_test(model, x)

    def test_maxpool_2d(self):
        model = torch.nn.MaxPool2d(5, padding=(1, 2))
        x = torch.randn(1, 20, 16, 50, requires_grad=True)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(8)
    def test_max_tensors(self):
        class MaxModel(torch.nn.Module):
            def forward(self, input, other):
                return torch.max(input, other)

        model = MaxModel()
        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 1, requires_grad=True)
        self.run_test(model, (x, y))

    def test_gt(self):
        class GreaterModel(torch.nn.Module):
            def forward(self, input, other):
                return input > other

        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        y = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.run_test(GreaterModel(), (x, y))

        x = torch.randint(10, (3, 4), dtype=torch.int32)
        y = torch.randint(10, (3, 4), dtype=torch.int32)
        self.run_test(GreaterModel(), (x, y))

    def test_gt_scalar(self):
        class GreaterModel(torch.nn.Module):
            def forward(self, input):
                return input > 1

        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.run_test(GreaterModel(), x)

        x = torch.randint(10, (3, 4), dtype=torch.int32)
        self.run_test(GreaterModel(), x)

    def test_lt(self):
        class LessModel(torch.nn.Module):
            def forward(self, input, other):
                return input > other

        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        y = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.run_test(LessModel(), (x, y))

        x = torch.randint(10, (3, 4), dtype=torch.int32)
        y = torch.randint(10, (3, 4), dtype=torch.int32)
        self.run_test(LessModel(), (x, y))

    def test_matmul(self):
        class MatmulModel(torch.nn.Module):
            def forward(self, input, other):
                return torch.matmul(input, other)

        x = torch.randn(3, 4, requires_grad=True)
        y = torch.randn(4, 5, requires_grad=True)
        self.run_test(MatmulModel(), (x, y))

        x = torch.randint(10, (3, 4))
        y = torch.randint(10, (4, 5))
        self.run_test(MatmulModel(), (x, y))

    def test_matmul_batch(self):
        class MatmulModel(torch.nn.Module):
            def forward(self, input, other):
                return torch.matmul(input, other)

        x = torch.randn(2, 3, 4, requires_grad=True)
        y = torch.randn(2, 4, 5, requires_grad=True)
        self.run_test(MatmulModel(), (x, y))

        x = torch.randint(10, (2, 3, 4))
        y = torch.randint(10, (2, 4, 5))
        self.run_test(MatmulModel(), (x, y))

    def test_view(self):
        class ViewModel(torch.nn.Module):
            def forward(self, input):
                return input.view(4, 24)

        x = torch.randint(10, (4, 2, 3, 4), dtype=torch.int32)
        self.run_test(ViewModel(), x)

    def test_flatten(self):
        class FlattenModel(torch.nn.Module):
            def forward(self, input):
                return torch.flatten(input)

        x = torch.randint(10, (1, 2, 3, 4))
        self.run_test(FlattenModel(), x)

    def test_flatten2d(self):
        class FlattenModel(torch.nn.Module):
            def forward(self, input):
                return torch.flatten(input, 1)

        x = torch.randint(10, (1, 2, 3, 4))
        self.run_test(FlattenModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_tensor_factories(self):
        class TensorFactory(torch.nn.Module):
            def forward(self, x):
                return torch.zeros(x.size()) + torch.ones(x.size())

        x = torch.randn(2, 3, 4)
        self.run_test(TensorFactory(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_tensor_factories_script(self):
        class TensorFactory(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                return torch.zeros(x.shape, dtype=torch.float) + torch.ones(x.shape, dtype=torch.float)

        x = torch.randn(2, 3, 4)
        self.run_test(TensorFactory(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_tensor_like_factories_script(self):
        class TensorFactory(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                zeros = torch.zeros_like(x, dtype=torch.float, layout=torch.strided, device=torch.device('cpu'))
                ones = torch.ones_like(x, dtype=torch.float, layout=torch.strided, device=torch.device('cpu'))
                return zeros + ones

        x = torch.randn(2, 3, 4)
        self.run_test(TensorFactory(), x)


# opset 7 tests
TestONNXRuntime_opset7 = type(str("TestONNXRuntime_opset7"),
                              (unittest.TestCase,),
                              dict(TestONNXRuntime.__dict__, opset_version=7))

# opset 8 tests
TestONNXRuntime_opset8 = type(str("TestONNXRuntime_opset8"),
                              (unittest.TestCase,),
                              dict(TestONNXRuntime.__dict__, opset_version=8))

# opset 10 tests
TestONNXRuntime_opset10 = type(str("TestONNXRuntime_opset10"),
                               (unittest.TestCase,),
                               dict(TestONNXRuntime.__dict__, opset_version=10))


if __name__ == '__main__':
    unittest.main()
