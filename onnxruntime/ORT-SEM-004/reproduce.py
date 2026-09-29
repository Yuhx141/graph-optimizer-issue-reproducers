#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import onnxruntime as ort

MODEL = Path(__file__).with_name("minimal_model.onnx")
DISABLED = ['SkipLayerNormFusion']

def load(level, disabled=()):
    so = ort.SessionOptions()
    so.graph_optimization_level = getattr(ort.GraphOptimizationLevel, level)
    so.intra_op_num_threads = so.inter_op_num_threads = 1
    return ort.InferenceSession(str(MODEL), so, providers=["CPUExecutionProvider"],
                                disabled_optimizers=set(disabled))

def make_feeds(session):
    dtypes = {"tensor(float)": np.float32, "tensor(float16)": np.float16,
              "tensor(double)": np.float64, "tensor(int64)": np.int64,
              "tensor(int32)": np.int32, "tensor(bool)": np.bool_}
    feeds = {}
    for x in session.get_inputs():
        shape = tuple(d if isinstance(d, int) and d > 0 else 2 for d in x.shape)
        dtype = dtypes[x.type]
        size = max(1, int(np.prod(shape or (1,))))
        if np.issubdtype(dtype, np.floating):
            value = np.linspace(-0.75, 0.75, size, dtype=dtype)
        elif dtype == np.bool_:
            value = np.ones(size, dtype=dtype)
        else:
            value = np.zeros(size, dtype=dtype)
        feeds[x.name] = value.reshape(shape)
    return feeds

reference_session = load("ORT_DISABLE_ALL")
feeds = make_feeds(reference_session)
reference = reference_session.run(None, feeds)
actual = load("ORT_ENABLE_ALL").run(None, feeds)
recovered = load("ORT_ENABLE_ALL", DISABLED).run(None, feeds)

def max_diff(left, right):
    diffs = []
    for a, b in zip(left, right):
        if a.shape != b.shape:
            return float("inf")
        diffs.append(float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))))
    return max(diffs, default=0.0)

enabled_diff = max_diff(reference, actual)
disabled_diff = max_diff(reference, recovered)
print("onnxruntime", ort.__version__)
print("enabled max abs diff:", enabled_diff)
print("pass-disabled max abs diff:", disabled_diff)
reproduced = enabled_diff > 1e-3 and disabled_diff <= 1e-3
print("REPRODUCED" if reproduced else "NOT REPRODUCED")
raise SystemExit(0 if reproduced else 1)
