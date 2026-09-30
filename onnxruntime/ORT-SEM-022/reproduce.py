#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).parent
FEEDS = {"X": np.array([[0.0, 1e-25, 0.0, 1e-25]], np.float64)}


def run(model, level, disabled=()):
    options = ort.SessionOptions()
    options.graph_optimization_level = level
    return ort.InferenceSession(
        str(ROOT / model),
        options,
        providers=["CPUExecutionProvider"],
        disabled_optimizers=set(disabled),
    ).run(None, FEEDS)[0]


cases = [
    ("layernorm.onnx", ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
     ["NoopElimination"], ["NoopElimination", "LayerNormFusionL1"]),
    ("simplified-layernorm.onnx", ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
     ["NoopElimination"], ["NoopElimination", "SimplifiedLayerNormFusion"]),
]

reproduced = True
for model, level, optimized_disabled, control_disabled in cases:
    baseline = run(model, ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
    optimized = run(model, level, optimized_disabled)
    control = run(model, level, control_disabled)
    case_reproduced = not np.array_equal(optimized, baseline) and np.array_equal(control, baseline)
    reproduced &= case_reproduced
    print(model, {"baseline": baseline.tolist(), "optimized": optimized.tolist(), "control": control.tolist()})

print("onnxruntime", ort.__version__)
print("REPRODUCED" if reproduced else "NOT REPRODUCED")
raise SystemExit(0 if reproduced else 1)
