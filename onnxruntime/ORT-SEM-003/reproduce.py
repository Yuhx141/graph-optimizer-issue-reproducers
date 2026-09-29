#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

MODEL = Path(__file__).with_name("minimal_model.onnx")

if "--worker" in sys.argv:
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = so.inter_op_num_threads = 1
    ort.InferenceSession(str(MODEL), so, providers=["CPUExecutionProvider"])
    raise SystemExit(0)

run = subprocess.run([sys.executable, __file__, "--worker"], capture_output=True, text=True)
print("worker return code:", run.returncode)
if run.stderr:
    print(run.stderr[-2000:])
reproduced = run.returncode in (-11, 139)
print("REPRODUCED" if reproduced else "NOT REPRODUCED")
raise SystemExit(0 if reproduced else 1)
