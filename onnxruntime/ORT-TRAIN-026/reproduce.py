#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import onnxruntime.capi._pybind_state as training


ROOT = Path(__file__).parent
X = np.arange(6, dtype=np.float32).reshape(2, 3)


def run(model):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    return session.run(None, {"X": X})[0]


def optimize(model):
    try:
        optimized = onnx.load_from_string(
            training.get_optimized_model(model.SerializeToString(), set(), ort.SessionOptions())
        )
        return "ok", optimized
    except Exception as exc:
        return "error", f"{type(exc).__name__}: {exc}"


omitted = onnx.load(ROOT / "transpose-omitted-perm.onnx")
explicit = onnx.load(ROOT / "transpose-explicit-default-perm.onnx")
omitted_source = run(omitted)
explicit_source = run(explicit)
omitted_status, omitted_result = optimize(omitted)
explicit_status, explicit_result = optimize(explicit)
explicit_output = run(explicit_result) if explicit_status == "ok" else None

reproduced = (
    np.array_equal(omitted_source, explicit_source)
    and omitted_status == "error"
    and "_Map_base::at" in omitted_result
    and explicit_status == "ok"
    and np.array_equal(omitted_source, explicit_output)
)
print("onnxruntime", ort.__version__)
print("source", omitted_source.tolist())
print("omitted perm", (omitted_status, omitted_result))
print("explicit default perm control", explicit_status, explicit_output.tolist())
print("REPRODUCED" if reproduced else "NOT REPRODUCED")
raise SystemExit(0 if reproduced else 1)
