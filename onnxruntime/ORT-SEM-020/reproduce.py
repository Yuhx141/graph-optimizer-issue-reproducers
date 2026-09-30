#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import onnxruntime as ort

MODEL = Path(__file__).with_name("minimal_model.onnx")
FEEDS = {"X": np.array([0.0, -1e-50, 1e-50], np.float64)}

def run(level, disabled=()):
    so = ort.SessionOptions()
    so.graph_optimization_level = level
    return ort.InferenceSession(str(MODEL), so, providers=["CPUExecutionProvider"],
                                disabled_optimizers=set(disabled)).run(None, FEEDS)[0]

baseline = run(ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
optimized = run(ort.GraphOptimizationLevel.ORT_ENABLE_BASIC)
control = run(ort.GraphOptimizationLevel.ORT_ENABLE_BASIC, ["NoopElimination"])
reproduced = not np.array_equal(optimized, baseline) and np.array_equal(control, baseline)
print("onnxruntime", ort.__version__)
print({"baseline": baseline.tolist(), "optimized": optimized.tolist(), "control": control.tolist()})
print("REPRODUCED" if reproduced else "NOT REPRODUCED")
raise SystemExit(0 if reproduced else 1)
