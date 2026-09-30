#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).parent
CONFIG = {"session.enable_quant_qdq_cleanup": "1"}
CASES = [
    ("qdq-first-q-public.onnx", {"X": np.array([-1.0, 0.0, 0.1, 1.0], np.float32)}, "producing output: Q"),
    ("qdq-shared-source-value.onnx", {"X": np.array([-1.0, 0.0, 0.1, 1.0], np.float32)}, "Node input 'R'"),
]


def run(model, feeds, level, disable_cleanup=False):
    options = ort.SessionOptions()
    options.graph_optimization_level = level
    if level != ort.GraphOptimizationLevel.ORT_DISABLE_ALL:
        for key, value in CONFIG.items():
            options.add_session_config_entry(key, value)
        if disable_cleanup:
            options.add_session_config_entry(
                "optimization.disable_specified_optimizers", "QDQFinalCleanupTransformer"
            )
    try:
        outputs = ort.InferenceSession(
            str(ROOT / model), options, providers=["CPUExecutionProvider"]
        ).run(None, feeds)
        return "ok", [value.tolist() for value in outputs]
    except Exception as exc:
        return "error", str(exc)


reproduced = True
for model, feeds, expected_error in CASES:
    baseline = run(model, feeds, ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
    optimized = run(model, feeds, ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED)
    control = run(model, feeds, ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED, True)
    case_reproduced = (
        baseline[0] == control[0] == "ok"
        and optimized[0] == "error"
        and expected_error in optimized[1]
    )
    reproduced &= case_reproduced
    print(model, {"baseline": baseline, "optimized": optimized, "control": control})

print("onnxruntime", ort.__version__)
print("REPRODUCED" if reproduced else "NOT REPRODUCED")
raise SystemExit(0 if reproduced else 1)
