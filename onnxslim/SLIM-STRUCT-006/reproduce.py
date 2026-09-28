#!/usr/bin/env python3
"""Reproduce FusionConvBN rewrites training-mode BatchNormalization outputs as Conv outputs."""

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import onnxslim

HERE = Path(__file__).resolve().parent
PASS = 'FusionConvBN'
EXPECTED = 'invalid_target'
DISABLE = {'skip_fusion_patterns': ['FusionConvBN']}
CONTROL_MUST_MATCH = True


def load_inputs(name):
    data = json.loads((HERE / name).read_text())
    return {key: np.asarray(item["values"], dtype=item["dtype"]).reshape(item["shape"])
            for key, item in data.items()}


def check(model):
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)


def run(model, feeds):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(model.SerializeToString(), options,
                                   providers=["CPUExecutionProvider"])
    selected = {item.name: feeds[item.name] for item in session.get_inputs()}
    return session.run(None, selected)


def same(left, right):
    return len(left) == len(right) and all(np.array_equal(a, b) for a, b in zip(left, right))


source = onnx.load(HERE / "source.onnx")
feeds = load_inputs("input.json")
check(source)
before = run(source, feeds)

# Positive control: disabling only the implicated pass restores the source result.
disabled = onnxslim.slim(onnx.load(HERE / "source.onnx"), **DISABLE)
check(disabled)
assert same(before, run(disabled, feeds))

# One-condition negative control: the adjacent graph must not reproduce the failure.
control = onnx.load(HERE / "control.onnx")
control_feeds = load_inputs("control-input.json")
check(control)
control_before = run(control, control_feeds)
control_after = onnxslim.slim(onnx.load(HERE / "control.onnx"))
check(control_after)
if CONTROL_MUST_MATCH:
    assert same(control_before, run(control_after, control_feeds))

error = None
after = None
try:
    optimized = onnxslim.slim(onnx.load(HERE / "source.onnx"))
    onnx.save(optimized, HERE / "optimized.onnx")
    check(optimized)
    after = run(optimized, feeds)
except Exception as exc:
    error = f"{type(exc).__name__}: {exc}"

print(f"pass: {PASS}")
print("source:", [value.tolist() for value in before])
print("optimized:", error or [value.tolist() for value in after])
print("pass disabled: matches source")
print("condition control: passes")

if EXPECTED == "semantic_mismatch":
    assert error is None and not same(before, after)
elif EXPECTED == "invalid_target":
    assert error is not None
else:
    assert error is not None
