#!/usr/bin/env python3
import subprocess, sys
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).parent
FEEDS = {
    "A": np.array([2., 4.], np.float32),
    "B": np.array([4., 8.], np.float32),
    "X": np.array([-3., 1.5], np.float32),
}

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

bad = ROOT / "scaled-sum-relu-third.onnx"
control = ROOT / "scaled-sum-identity-third-control.onnx"
bad_opt, bad_nodes = optimize(bad.name)
control_opt, control_nodes = optimize(control.name)
source, optimized = run(bad), run(bad_opt)
control_source, control_optimized = run(control), run(control_opt)
reproduced = (
    np.array_equal(source, np.array([2., 5.5], np.float32))
    and np.array_equal(optimized, np.array([-1., 5.5], np.float32))
    and np.array_equal(control_source, control_optimized)
    and "com.microsoft::ScaledSum" in bad_nodes
)
print("onnxruntime", ort.__version__, ort.get_available_providers())
print("Relu source/optimized", source.tolist(), optimized.tolist())
print("Identity control", control_source.tolist(), control_optimized.tolist())
print("optimized nodes", bad_nodes)
print("REPRODUCED" if reproduced else "NOT REPRODUCED")
raise SystemExit(0 if reproduced else 1)
