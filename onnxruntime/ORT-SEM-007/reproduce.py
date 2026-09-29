#!/usr/bin/env python3
import tempfile
from pathlib import Path
import onnxruntime as ort

MODEL = Path(__file__).with_name("minimal_model.onnx")
DISABLED = ['GatherToSliceFusion']

def options(level, optimized=None):
    value = ort.SessionOptions()
    value.graph_optimization_level = getattr(ort.GraphOptimizationLevel, level)
    value.intra_op_num_threads = value.inter_op_num_threads = 1
    if optimized is not None:
        value.optimized_model_filepath = str(optimized)
    return value

providers = ["CPUExecutionProvider"]
ort.InferenceSession(str(MODEL), options("ORT_DISABLE_ALL"), providers=providers)
recovered = False
try:
    ort.InferenceSession(str(MODEL), options("ORT_ENABLE_ALL"), providers=providers,
                         disabled_optimizers=set(DISABLED))
    recovered = True
except Exception as error:
    print("pass-disabled control failed:", error)

with tempfile.TemporaryDirectory() as temp:
    target = Path(temp) / "optimized.onnx"
    try:
        ort.InferenceSession(str(MODEL), options("ORT_ENABLE_ALL", target), providers=providers)
        try:
            ort.InferenceSession(str(target), options("ORT_DISABLE_ALL"), providers=providers)
            observed = "optimized model unexpectedly reloaded"
            failed = False
        except Exception as error:
            observed = f"optimized model reload failed: {error}"
            failed = True
    except Exception as error:
        observed = f"optimization/session creation failed: {error}"
        failed = True

print("onnxruntime", ort.__version__)
print(observed)
reproduced = failed and recovered
print("REPRODUCED" if reproduced else "NOT REPRODUCED")
raise SystemExit(0 if reproduced else 1)
