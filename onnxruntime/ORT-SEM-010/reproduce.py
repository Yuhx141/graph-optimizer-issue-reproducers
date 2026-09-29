#!/usr/bin/env python3
import tempfile
from pathlib import Path
import onnxruntime as ort

MODEL = Path(__file__).with_name("minimal_model.onnx")
DISABLED = ['ConstantSharing']

def load(model, level, disabled=(), optimized=None):
    so = ort.SessionOptions()
    so.graph_optimization_level = getattr(ort.GraphOptimizationLevel, level)
    so.intra_op_num_threads = so.inter_op_num_threads = 1
    if optimized is not None:
        so.optimized_model_filepath = str(optimized)
    return ort.InferenceSession(str(model), so, providers=["CPUExecutionProvider"],
                                disabled_optimizers=set(disabled))

reference = [x.name for x in load(MODEL, "ORT_DISABLE_ALL").get_inputs()]
with tempfile.TemporaryDirectory() as temp:
    target = Path(temp) / "optimized.onnx"
    recovered = Path(temp) / "disabled.onnx"
    load(MODEL, "ORT_ENABLE_ALL", optimized=target)
    load(MODEL, "ORT_ENABLE_ALL", DISABLED, recovered)
    actual = [x.name for x in load(target, "ORT_DISABLE_ALL").get_inputs()]
    disabled = [x.name for x in load(recovered, "ORT_DISABLE_ALL").get_inputs()]
added = [x for x in actual if x not in reference]
print("onnxruntime", ort.__version__)
print("reference inputs:", reference)
print("optimized inputs:", actual)
print("pass-disabled inputs:", disabled)
print("new required inputs:", added)
reproduced = bool(added) and disabled == reference
print("REPRODUCED" if reproduced else "NOT REPRODUCED")
raise SystemExit(0 if reproduced else 1)
