#!/usr/bin/env python3
"""Reproduce Pad_Fusion changes zero padding semantics before MaxPool for negative inputs."""

import json
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
PASS = 'Pad_Fusion'
EXPECTED = 'semantic_mismatch'


def load_inputs(name):
    data = json.loads((HERE / name).read_text())
    return {key: np.asarray(item["values"], dtype=item["dtype"]).reshape(item["shape"])
            for key, item in data.items()}


def check(path):
    model = onnx.load(path)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)


def run(source, feeds, optimized=None, disabled=None):
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    if optimized is None:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    else:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.optimized_model_filepath = str(optimized)
    session = ort.InferenceSession(str(source), options,
                                   providers=["CPUExecutionProvider"],
                                   disabled_optimizers=disabled)
    selected = {item.name: feeds[item.name] for item in session.get_inputs()}
    return session.run(None, selected)


def same(left, right):
    return len(left) == len(right) and all(np.array_equal(a, b) for a, b in zip(left, right))


feeds = load_inputs("input.json")
check(HERE / "source.onnx")
before = run(HERE / "source.onnx", feeds)

with tempfile.TemporaryDirectory() as directory:
    directory = Path(directory)

    # Positive control: disabling only the implicated transformer restores the result.
    disabled_path = directory / "disabled.onnx"
    disabled = run(HERE / "source.onnx", feeds, disabled_path, {PASS})
    check(disabled_path)
    assert same(before, disabled)

    # One-condition negative control.
    control_feeds = load_inputs("control-input.json")
    control_before = run(HERE / "control.onnx", control_feeds)
    control_path = directory / "control-optimized.onnx"
    control_after = run(HERE / "control.onnx", control_feeds, control_path)
    check(control_path)
    assert same(control_before, control_after)

    optimized_path = directory / "optimized.onnx"
    error = None
    after = None
    try:
        after = run(HERE / "source.onnx", feeds, optimized_path)
        check(optimized_path)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

print(f"transformer: {PASS}")
print("source:", [value.tolist() for value in before])
print("optimized:", error or [value.tolist() for value in after])
print("transformer disabled: matches source")
print("condition control: passes")

if EXPECTED == "semantic_mismatch":
    assert error is None and not same(before, after)
else:
    assert error is not None
