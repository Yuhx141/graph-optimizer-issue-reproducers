#!/usr/bin/env python3
import subprocess
import sys
import os
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


ROOT = Path(__file__).parent
PROVIDER = os.environ.get("ORT_PROVIDER", "CPUExecutionProvider")
FEEDS = {
    "A": np.array([2.0, 4.0], np.float32),
    "B": np.array([4.0, 8.0], np.float32),
    "X": np.array([0.5, 1.5], np.float32),
}


def child(path):
    import onnxruntime.capi._pybind_state as training

    model = onnx.load(path)
    optimized = onnx.load_from_string(
        training.get_optimized_model(model.SerializeToString(), set(), ort.SessionOptions())
    )
    onnx.save(optimized, path.with_suffix(".optimized.onnx"))


if len(sys.argv) == 3 and sys.argv[1] == "--child":
    child(Path(sys.argv[2]))
    raise SystemExit(0)


def run(path):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    if PROVIDER == "CUDAExecutionProvider":
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(str(path), options, providers=[PROVIDER])
    return session.run(None, FEEDS)[0]


def optimize(path):
    return subprocess.run(
        [sys.executable, __file__, "--child", str(path)], capture_output=True, text=True
    )


direct = ROOT / "scaled-sum-graph-input.onnx"
control = ROOT / "scaled-sum-identity-control.onnx"
direct_source = run(direct)
control_source = run(control)
direct_result = optimize(direct)
control_result = optimize(control)
control_optimized_path = control.with_suffix(".optimized.onnx")
control_nodes = (
    [node.op_type for node in onnx.load(control_optimized_path).graph.node]
    if control_result.returncode == 0
    else []
)
control_optimized_path.unlink(missing_ok=True)

reproduced = (
    np.array_equal(direct_source, control_source)
    and direct_result.returncode in (-11, 139)
    and control_result.returncode == 0
    and "ScaledSum" in control_nodes
)
print("onnxruntime", ort.__version__)
print("provider", PROVIDER)
print("source", direct_source.tolist())
print("direct graph input optimizer return code", direct_result.returncode)
print("Identity control optimizer return code", control_result.returncode)
print("Identity control optimized nodes", control_nodes)
print("REPRODUCED" if reproduced else "NOT REPRODUCED")
raise SystemExit(0 if reproduced else 1)
