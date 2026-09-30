#!/usr/bin/env python3
import subprocess, sys
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).parent
FEEDS = {
    "scores": np.array([[1., 2., 3.], [3., 1., 0.]], np.float32),
    "labels": np.array([2, 0], np.int64),
}

if len(sys.argv) == 3 and sys.argv[1] == "--optimize":
    import onnxruntime.capi._pybind_state as training
    source = ROOT / sys.argv[2]
    data = training.get_optimized_model(onnx.load(source).SerializeToString(), set(), ort.SessionOptions())
    onnx.save(onnx.load_from_string(data), source.with_suffix(".optimized.onnx"))
    raise SystemExit(0)

def optimize(name):
    return subprocess.run([sys.executable, __file__, "--optimize", name], capture_output=True, text=True)

def run(path):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(str(path), options, providers=["CUDAExecutionProvider"])
    return session.run(None, FEEDS)

direct = ROOT / "sce-direct-graph-input.onnx"
control = ROOT / "sce-identity-control.onnx"
direct_source = run(direct)
control_source = run(control)
direct_result = optimize(direct.name)
control_result = optimize(control.name)
control_opt = control.with_suffix(".optimized.onnx")
control_optimized = run(control_opt) if control_result.returncode == 0 else []
reproduced = (
    all(np.array_equal(a, b) for a, b in zip(direct_source, control_source))
    and direct_result.returncode in (-11, 139)
    and control_result.returncode == 0
    and all(np.array_equal(a, b) for a, b in zip(control_source, control_optimized))
)
print("onnxruntime", ort.__version__, ort.get_available_providers())
print("direct optimizer return code", direct_result.returncode)
print("Identity control optimizer return code", control_result.returncode)
print("REPRODUCED" if reproduced else "NOT REPRODUCED")
raise SystemExit(0 if reproduced else 1)
