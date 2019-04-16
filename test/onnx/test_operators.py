from test_pytorch_common import run_tests

import glob
import os
import shutil
import common_utils as common

'''Usage: python test/onnx/test_operators.py [--no-onnx] [--produce-onnx-test-data]
          --no-onnx: no onnx python dependence
          --produce-onnx-test-data: generate onnx test data
'''

from test_operators_opset9 import TestOperators_opset9
from test_operators_opset10 import TestOperators_opset10


if __name__ == '__main__':
    no_onnx_dep_flag = '--no-onnx'
    _onnx_dep = no_onnx_dep_flag not in common.UNITTEST_ARGS
    if no_onnx_dep_flag in common.UNITTEST_ARGS:
        common.UNITTEST_ARGS.remove(no_onnx_dep_flag)
    onnx_test_flag = '--produce-onnx-test-data'
    _onnx_test = onnx_test_flag in common.UNITTEST_ARGS
    if onnx_test_flag in common.UNITTEST_ARGS:
        common.UNITTEST_ARGS.remove(onnx_test_flag)
    if _onnx_test:
        _onnx_dep = True
        import test_onnx_common
        for d in glob.glob(os.path.join(test_onnx_common.pytorch_operator_dir, "test_operator_*")):
            shutil.rmtree(d)
    run_tests()
