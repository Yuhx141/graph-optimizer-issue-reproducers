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
    result = subprocess.run([sys.executable, __file__, "--optimize", name], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr)
    path = (ROOT / name).with_suffix(".optimized.onnx")
    model = onnx.load(path)
    try:
        onnx.checker.check_model(model)
        checker = "ok"
    except Exception as exc:
        checker = str(exc)
    return path, checker, [f"{n.domain + '::' if n.domain else ''}{n.op_type}" for n in model.graph.node]

def run(path):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(str(path), options, providers=["CUDAExecutionProvider"])
    return session.run(None, FEEDS)

axis0 = ROOT / "nll-logsoftmax-axis-zero.onnx"
axis1 = ROOT / "nll-logsoftmax-axis-one-control.onnx"
public = ROOT / "nll-public-log-prob.onnx"
axis0_opt, axis0_checker, axis0_nodes = optimize(axis0.name)
axis1_opt, axis1_checker, axis1_nodes = optimize(axis1.name)
public_opt, public_checker, public_nodes = optimize(public.name)
axis0_source = run(axis0)[0]
axis0_optimized = run(axis0_opt)[0]
axis1_source = run(axis1)[0]
axis1_optimized = run(axis1_opt)[0]
public_source = run(public)
try:
    run(public_opt)
    public_load_error = ""
except Exception as exc:
    public_load_error = str(exc)
reproduced = (
    not np.array_equal(axis0_source, axis0_optimized)
    and np.array_equal(axis1_source, axis1_optimized)
    and axis0_checker == axis1_checker == "ok"
    and public_checker != "ok"
    and "log_prob" in public_checker
    and "log_prob" in public_load_error
    and "com.microsoft::SoftmaxCrossEntropyLossInternal" in axis0_nodes
)
print("onnxruntime", ort.__version__, ort.get_available_providers())
print("axis=0 source/optimized", axis0_source.tolist(), axis0_optimized.tolist())
print("axis=1 control", axis1_source.tolist(), axis1_optimized.tolist())
print("public output checker", public_checker)
print("public output load", public_load_error)
print("REPRODUCED" if reproduced else "NOT REPRODUCED")
raise SystemExit(0 if reproduced else 1)
