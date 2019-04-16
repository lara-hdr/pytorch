import torch 

import test_pytorch_onnx_caffe2_helper
from test_pytorch_onnx_caffe2_helper import TestCaffe2Backend

BATCH_SIZE = test_pytorch_onnx_caffe2_helper.BATCH_SIZE
RNN_BATCH_SIZE = test_pytorch_onnx_caffe2_helper.RNN_BATCH_SIZE
RNN_SEQUENCE_LENGTH = test_pytorch_onnx_caffe2_helper.RNN_SEQUENCE_LENGTH
RNN_INPUT_SIZE = test_pytorch_onnx_caffe2_helper.RNN_INPUT_SIZE
RNN_HIDDEN_SIZE = test_pytorch_onnx_caffe2_helper.RNN_HIDDEN_SIZE
model_urls = test_pytorch_onnx_caffe2_helper.model_urls


class TestCaffe2Backend_opset10(TestCaffe2Backend):
    def test_isnan(self):
        class IsNaNModel(torch.nn.Module):
            def forward(self, input):
                return torch.isnan(input)

        x = torch.tensor([1.0, float('nan'), 2.0])
        self.run_model_test(IsNaNModel(), train=False, input=x, batch_size=BATCH_SIZE, use_gpu=False, opset_version=10)
