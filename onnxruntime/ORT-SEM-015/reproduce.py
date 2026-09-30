#!/usr/bin/env python3
import tempfile
from pathlib import Path
import onnxruntime as ort

MODEL = Path(__file__).with_name("minimal_model.onnx")
DISABLED = ['AttentionFusion']

def options(level, optimized=None):
    so = ort.SessionOptions()
    so.graph_optimization_level = getattr(ort.GraphOptimizationLevel, level)
    so.intra_op_num_threads = so.inter_op_num_threads = 1
    if optimized is not None:
        so.optimized_model_filepath = str(optimized)
    return so

providers = ["CPUExecutionProvider"]
ort.InferenceSession(str(MODEL), options("ORT_DISABLE_ALL"), providers=providers)
with tempfile.TemporaryDirectory() as temp:
    target = Path(temp) / "optimized.onnx"
    try:
        ort.InferenceSession(str(MODEL), options("ORT_ENABLE_ALL", target), providers=providers)
        try:
            ort.InferenceSession(str(target), options("ORT_DISABLE_ALL"), providers=providers)
            observed = "optimized model unexpectedly reloaded"
            reproduced = False
        except Exception as error:
            observed = f"optimized model reload failed: {error}"
            reproduced = True
    except Exception as error:
        observed = f"optimization/session creation failed: {error}"
        reproduced = True
    ort.InferenceSession(str(MODEL), options("ORT_ENABLE_ALL"), providers=providers,
                         disabled_optimizers=set(DISABLED))
print("onnxruntime", ort.__version__)
print(observed)
print("REPRODUCED" if reproduced else "NOT REPRODUCED")
raise SystemExit(0 if reproduced else 1)
