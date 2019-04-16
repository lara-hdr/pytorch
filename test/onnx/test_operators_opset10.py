import torch
from test_operators_helper import TestOperators


class TestOperators_opset10(TestOperators):

    def test_master_opset(self):
        x = torch.randn(2, 3).float()
        y = torch.randn(2, 3).float()
        self.assertONNX(lambda x, y: x + y, (x, y), opset_version=10)
