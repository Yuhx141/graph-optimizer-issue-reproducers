#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import onnxruntime as ort

MODEL = Path(__file__).with_name("minimal_model.onnx")
FEEDS = {
    "A": np.array([[1.0, 2.0], [3.0, 4.0]], np.float64),
    "B": np.array([[5.0, 6.0], [7.0, 8.0]], np.float64),
}

def run(level, disabled=()):
    so = ort.SessionOptions()
    so.graph_optimization_level = level
    return ort.InferenceSession(str(MODEL), so, providers=["CPUExecutionProvider"],
                                disabled_optimizers=set(disabled)).run(None, FEEDS)[0]

baseline = run(ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
optimized = run(ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED, ["NoopElimination"])
control = run(ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED, ["NoopElimination", "MatMulScaleFusion"])
reproduced = not np.array_equal(optimized, baseline) and np.array_equal(control, baseline)
print("onnxruntime", ort.__version__)
print({"baseline": baseline.tolist(), "optimized": optimized.tolist(), "control": control.tolist()})
print("REPRODUCED" if reproduced else "NOT REPRODUCED")
raise SystemExit(0 if reproduced else 1)
