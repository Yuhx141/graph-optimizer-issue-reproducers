#!/usr/bin/env python3
"""Reproduce Pad-to-MaxPool fusion does not preserve the indices output coordinate system."""

import json
from pathlib import Path

import numpy as np
import onnx
import onnxoptimizer
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
PASS = 'fuse_pad_into_pool'
EXPECTED = 'semantic_mismatch'


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

# Positive control: running no optimizer pass must preserve the source result.
disabled = onnxoptimizer.optimize(onnx.load(HERE / "source.onnx"), [])
check(disabled)
assert same(before, run(disabled, feeds))

# One-condition negative control: the same pass must not trigger the reported failure.
control = onnx.load(HERE / "control.onnx")
control_feeds = load_inputs("control-input.json")
check(control)
control_before = run(control, control_feeds)
control_after = onnxoptimizer.optimize(onnx.load(HERE / "control.onnx"), [PASS])
check(control_after)
assert same(control_before, run(control_after, control_feeds))

optimizer_error = None
runtime_error = None
validation_error = None
optimized = None
after = None
try:
    optimized = onnxoptimizer.optimize(onnx.load(HERE / "source.onnx"), [PASS])
    onnx.save(optimized, HERE / "optimized.onnx")
except Exception as exc:
    optimizer_error = f"{type(exc).__name__}: {exc}"

if optimized is not None:
    try:
        after = run(optimized, feeds)
    except Exception as exc:
        runtime_error = f"{type(exc).__name__}: {exc}"
    try:
        check(optimized)
    except Exception as exc:
        validation_error = f"{type(exc).__name__}: {exc}"

print(f"pass: {PASS}")
print("source:", [value.tolist() for value in before])
print("optimized:", optimizer_error or runtime_error or [value.tolist() for value in after])
if validation_error:
    print("optimized validation:", validation_error)
print("pass disabled: matches source")
print("condition control: passes")

if EXPECTED == "semantic_mismatch":
    assert optimizer_error is None and runtime_error is None and not same(before, after)
elif EXPECTED == "invalid_target":
    assert optimizer_error is None and (validation_error is not None or runtime_error is not None)
else:
    assert optimizer_error is not None
