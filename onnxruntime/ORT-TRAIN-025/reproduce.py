#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import onnxruntime.capi._pybind_state as training


ROOT = Path(__file__).parent
X = np.array([-2.0, -1.0, 1.0, 2.0], np.float32)


def run(model, feeds):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    try:
        session = ort.InferenceSession(
            model.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        return "ok", [value.tolist() for value in session.run(None, feeds)]
    except Exception as exc:
        return "error", str(exc)


def optimize(model):
    try:
        return "ok", onnx.load_from_string(
            training.get_optimized_model(model.SerializeToString(), set(), ort.SessionOptions())
        )
    except Exception as exc:
        return "error", str(exc)


public_model = onnx.load(ROOT / "public-q.onnx")
public_source = run(public_model, {"X": X})
public_status, public_optimized = optimize(public_model)
if public_status == "ok":
    try:
        onnx.checker.check_model(public_optimized)
        public_checker = "ok"
    except Exception as exc:
        public_checker = str(exc)
    public_runtime = run(public_optimized, {"X": X})
else:
    public_checker = public_runtime = public_optimized

shared_model = onnx.load(ROOT / "shared-zero-point.onnx")
shared_source = run(shared_model, {"X1": X, "X2": -X})
shared_status, shared_optimized = optimize(shared_model)

reproduced = (
    public_source[0] == shared_source[0] == "ok"
    and public_status == "ok"
    and "Graph output 'Q'" in public_checker
    and public_runtime[0] == "error"
    and shared_status == "error"
    and "zero point initializer with name zero" in shared_optimized
)

print("onnxruntime", ort.__version__)
print("public Q", {"source": public_source, "checker": public_checker, "optimized": public_runtime})
print("shared zero point", {"source": shared_source, "optimized": (shared_status, shared_optimized)})
print("REPRODUCED" if reproduced else "NOT REPRODUCED")
raise SystemExit(0 if reproduced else 1)
