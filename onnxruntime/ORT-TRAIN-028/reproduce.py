#!/usr/bin/env python3
import subprocess, sys
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).parent
FEEDS = {"A": np.array([[np.inf, 0, 0, 0]], np.float16)}

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
    return session.run(None, FEEDS)[0].tolist()

bad = ROOT / "isinf-threshold-one.onnx"
control = ROOT / "isinf-threshold-zero-control.onnx"
bad_opt, bad_nodes = optimize(bad.name)
control_opt, control_nodes = optimize(control.name)
values = {
    "threshold=1 source": run(bad),
    "threshold=1 optimized": run(bad_opt),
    "threshold=0 source": run(control),
    "threshold=0 optimized": run(control_opt),
}
reproduced = (
    values["threshold=1 source"] is False
    and values["threshold=1 optimized"] is True
    and values["threshold=0 source"] is values["threshold=0 optimized"] is True
    and "com.microsoft::IsAllFinite" in bad_nodes and "Not" in bad_nodes
)
print("onnxruntime", ort.__version__, ort.get_available_providers())
print(values)
print("optimized nodes", bad_nodes)
print("REPRODUCED" if reproduced else "NOT REPRODUCED")
raise SystemExit(0 if reproduced else 1)
