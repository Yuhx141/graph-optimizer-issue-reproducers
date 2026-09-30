#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import onnxruntime as ort

MODEL = Path(__file__).with_name("minimal_model.onnx")
FEEDS = {"X": np.full(1024, 0.5, np.float32)}

def run(level, disabled=()):
    so = ort.SessionOptions()
    so.graph_optimization_level = level
    session = ort.InferenceSession(str(MODEL), so, providers=["CPUExecutionProvider"],
                                   disabled_optimizers=set(disabled))
    return [int(np.count_nonzero(session.run(None, FEEDS)[0])) for _ in range(3)]

baseline = run(ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
optimized = run(ort.GraphOptimizationLevel.ORT_ENABLE_BASIC)
control = run(ort.GraphOptimizationLevel.ORT_ENABLE_BASIC, ["CommonSubexpressionElimination"])
reproduced = all(v > 0 for v in baseline + control) and optimized == [0, 0, 0]
print("onnxruntime", ort.__version__)
print("nonzero counts:", {"baseline": baseline, "optimized": optimized, "control": control})
print("REPRODUCED" if reproduced else "NOT REPRODUCED")
raise SystemExit(0 if reproduced else 1)
