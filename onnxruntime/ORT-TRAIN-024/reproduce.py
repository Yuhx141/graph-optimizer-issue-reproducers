#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import onnxruntime.capi._pybind_state as training
from onnx import helper


ROOT = Path(__file__).parent
X = np.array([-20.0, -1.0, 0.0, 1.0, 20.0], np.float32)


def optimize(path):
    model = onnx.load(ROOT / path)
    return onnx.load_from_string(
        training.get_optimized_model(model.SerializeToString(), set(), ort.SessionOptions())
    )


def run(model):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    return session.run(None, {"X": X})[0]


default_model = onnx.load(ROOT / "qdq-default-zero-point.onnx")
explicit_model = onnx.load(ROOT / "qdq-explicit-uint8-zero-point.onnx")
default_optimized = optimize("qdq-default-zero-point.onnx")
explicit_optimized = optimize("qdq-explicit-uint8-zero-point.onnx")

baseline = run(default_model)
same_semantics_baseline = run(explicit_model)
wrong = run(default_optimized)
control = run(explicit_optimized)


def attributes(model):
    return {
        attr.name: helper.get_attribute_value(attr)
        for node in model.graph.node
        for attr in node.attribute
    }


default_attributes = attributes(default_optimized)
control_attributes = attributes(explicit_optimized)
reproduced = (
    np.array_equal(baseline, same_semantics_baseline)
    and not np.array_equal(baseline, wrong)
    and np.array_equal(baseline, control)
    and default_attributes.get("quant_min") == -128
    and default_attributes.get("quant_max") == 127
    and control_attributes.get("quant_min") == 0
    and control_attributes.get("quant_max") == 255
)

print("onnxruntime", ort.__version__)
print("input", X.tolist())
print("source/default", baseline.tolist())
print("optimized/default", wrong.tolist(), default_attributes)
print("optimized/explicit UINT8 control", control.tolist(), control_attributes)
print("REPRODUCED" if reproduced else "NOT REPRODUCED")
raise SystemExit(0 if reproduced else 1)
