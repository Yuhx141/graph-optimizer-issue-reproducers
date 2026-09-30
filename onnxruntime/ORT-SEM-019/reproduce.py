#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import onnxruntime as ort

MODEL = Path(__file__).with_name("minimal_model.onnx")
FEEDS = {"X": np.array([2.0], np.float32)}

def run(level, disabled=()):
    so = ort.SessionOptions()
    so.graph_optimization_level = level
    return ort.InferenceSession(str(MODEL), so, providers=["CPUExecutionProvider"],
                                disabled_optimizers=set(disabled)).run(None, FEEDS)[0]

baseline = run(ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
optimized = run(ort.GraphOptimizationLevel.ORT_ENABLE_BASIC)
control = run(ort.GraphOptimizationLevel.ORT_ENABLE_BASIC, ["ExpandElimination"])
reproduced = baseline.shape == control.shape == (0,) and optimized.shape == (1,) and optimized.tolist() == [-2.0]
print("onnxruntime", ort.__version__)
print({"baseline_shape": baseline.shape, "optimized_shape": optimized.shape, "control_shape": control.shape})
print("REPRODUCED" if reproduced else "NOT REPRODUCED")
raise SystemExit(0 if reproduced else 1)
