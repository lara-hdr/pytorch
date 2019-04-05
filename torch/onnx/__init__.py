import torch._C as _C

TensorProtoDataType = _C._onnx.TensorProtoDataType
OperatorExportTypes = _C._onnx.OperatorExportTypes
PYTORCH_ONNX_CAFFE2_BUNDLE = _C._onnx.PYTORCH_ONNX_CAFFE2_BUNDLE

ONNX_ARCHIVE_MODEL_PROTO_NAME = "__MODEL_PROTO"


class SymbolicRegistry:
    # from collections import defaultdict
    def SymbolicRegistry(self, domain, version):
        _registry = {}  # defaultdict(dict)
        self.load(opset, domain)

    def load(self, domain, version):
        self.set(domain, version)
        if not self.is_loaded(domain, version):
            self._load()

    def _load():
        # for opname in dir(_symblic_versions[_version]):
        #    _registry[domain][_version][opname] = _symblic_versions[_version].opname
        it_version = _version  # - 1
        while it_opset >= 9:
            for opname in dir(_symblic_versions[it_version]):
                if _registry[domain][_version][opname] is None:
                    _registry[domain][_version][opname] = _symblic_versions[it_version].opname
            it_version = it_version - 1

    def set(self, domain, version):
        if _symblic_versions[_version] is None:  # or domain not supported
            warnings.warn("ONNX export failed. Opset version {} is not supported".format(version))
        _opset = opset
        _domain = domain

    def is_loaded(self, opset, domain):
        return _registry[domain][version] is not None

    def is_exportable(self, opname):
        return _registry[domain][version][opname]

    def get_op(self):
        return _registry[domain][version]

    _symblic_versions = {
        9 : torch.onnx.symbolic_opset9,
        10 : torch.onnx.symbolic_opset10
    }

symbolic_registry = None


class ExportTypes:
    PROTOBUF_FILE = 1
    ZIP_ARCHIVE = 2
    COMPRESSED_ZIP_ARCHIVE = 3
    DIRECTORY = 4


def _export(*args, **kwargs):
    from torch.onnx import utils
    return utils._export(*args, **kwargs)


def export(*args, **kwargs):
    from torch.onnx import utils
    return utils.export(*args, **kwargs)


def export_to_pretty_string(*args, **kwargs):
    from torch.onnx import utils
    return utils.export_to_pretty_string(*args, **kwargs)


def _export_to_pretty_string(*args, **kwargs):
    from torch.onnx import utils
    return utils._export_to_pretty_string(*args, **kwargs)


def _optimize_trace(trace, operator_export_type):
    from torch.onnx import utils
    trace.set_graph(utils._optimize_graph(trace.graph(), operator_export_type))


def set_training(*args, **kwargs):
    from torch.onnx import utils
    return utils.set_training(*args, **kwargs)


def _run_symbolic_function(*args, **kwargs):
    from torch.onnx import utils
    return utils._run_symbolic_function(*args, **kwargs)


def _run_symbolic_method(*args, **kwargs):
    from torch.onnx import utils
    return utils._run_symbolic_method(*args, **kwargs)
