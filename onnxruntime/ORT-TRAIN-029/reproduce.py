#!/usr/bin/env python3
import subprocess, sys
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).parent
FEEDS = {"X": np.array([-0.2, 0.15, 0.25, 1.05], np.float32), "dq_zero": np.array(0, np.uint8)}

if len(sys.argv) == 3 and sys.argv[1] == "--optimize":
    import onnxruntime.capi._pybind_state as training
    source = ROOT / sys.argv[2]
    data = training.get_optimized_model(onnx.load(source).SerializeToString(), set(), ort.SessionOptions())
    onnx.save(onnx.load_from_string(data), source.with_suffix(".optimized.onnx"))
    raise SystemExit(0)

def optimize(name):
    result = subprocess.run([sys.executable, __file__, "--optimize", name], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr)
    path = (ROOT / name).with_suffix(".optimized.onnx")
    model = onnx.load(path)
    onnx.checker.check_model(model)
    return path, [f"{n.domain + '::' if n.domain else ''}{n.op_type}" for n in model.graph.node]

def run(path):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(str(path), options, providers=["CUDAExecutionProvider"])
    return session.run(None, FEEDS)[0]

bad = ROOT / "qdq-mismatched-scale.onnx"
control = ROOT / "qdq-equal-scale-control.onnx"
bad_opt, bad_nodes = optimize(bad.name)
control_opt, control_nodes = optimize(control.name)
source = run(bad)
optimized = run(bad_opt)
control_source = run(control)
control_optimized = run(control_opt)
reproduced = (
    np.array_equal(source, np.array([0, .4, .4, 2], np.float32))
    and np.array_equal(optimized, np.array([0, .2, .2, 1], np.float32))
    and np.array_equal(control_source, control_optimized)
    and "com.microsoft::FakeQuant" in bad_nodes
)
print("onnxruntime", ort.__version__, ort.get_available_providers())
print("mismatched source", source.tolist())
print("mismatched optimized", optimized.tolist())
print("equal-scale control", control_source.tolist(), control_optimized.tolist())
print("optimized nodes", bad_nodes)
print("REPRODUCED" if reproduced else "NOT REPRODUCED")
raise SystemExit(0 if reproduced else 1)
