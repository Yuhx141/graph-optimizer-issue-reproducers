#!/usr/bin/env python3
"""Self-contained runner for an onnxsim issue attachment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASES = json.loads((HERE / "cases.json").read_text())


def run_model(model, feeds):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    values = session.run(None, {item.name: feeds[item.name] for item in session.get_inputs()})
    return {item.name: value for item, value in zip(session.get_outputs(), values)}


def compare(left, right):
    import numpy as np

    if left.keys() != right.keys():
        return {"same_exact": False, "finite_mask_equal": False, "max_abs": None, "output_names_differ": True}
    result = {"same_exact": True, "finite_mask_equal": True, "max_abs": 0.0}
    for name in left:
        a, b = np.asarray(left[name]), np.asarray(right[name])
        if a.shape != b.shape or a.dtype != b.dtype:
            return {"same_exact": False, "finite_mask_equal": False, "max_abs": None, "shape_or_dtype_differ": True}
        result["same_exact"] &= bool(np.array_equal(a, b, equal_nan=True))
        if np.issubdtype(a.dtype, np.floating):
            result["finite_mask_equal"] &= bool(np.array_equal(np.isfinite(a), np.isfinite(b)))
            joint = np.isfinite(a) & np.isfinite(b)
            if joint.any():
                result["max_abs"] = max(result["max_abs"], float(np.max(np.abs(a[joint].astype(np.float64) - b[joint].astype(np.float64)))))
    return result


def worker(index: int, mode: str) -> int:
    import numpy as np
    import onnx
    import onnxsim

    case = CASES[index]
    source = onnx.load(HERE / case["source"])
    row = {"case": case["name"], "mode": mode, "pass": case["pass"], "onnxsim_version": getattr(onnxsim, "__version__", "unknown")}
    try:
        onnx.checker.check_model(source, full_check=True)
        row["source_checker"] = "passed"
        kwargs = {"skipped_optimizers": [case["pass"]]} if mode == "disabled" else {}
        target, check_ok = onnxsim.simplify(onnx.load_from_string(source.SerializeToString()), check_n=0, **kwargs)
        row.update(check_ok=bool(check_ok), target_ops=[node.op_type for node in target.graph.node])
        try:
            onnx.checker.check_model(target, full_check=True)
            row["target_checker"] = "passed"
        except Exception as exc:
            row["target_checker"] = f"{type(exc).__name__}: {exc}"
        if case.get("input"):
            with np.load(HERE / case["input"], allow_pickle=False) as archive:
                feeds = dict(archive)
            baseline = run_model(source, feeds)
            try:
                observed = run_model(target, feeds)
                row["runtime"] = "passed"
                row["comparison"] = compare(baseline, observed)
            except Exception as exc:
                row["runtime"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        row["simplify_error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(row, sort_keys=True), flush=True)
    return 0


def classify(process, payload):
    if process.returncode != 0:
        return "process_crash"
    if "simplify_error" in payload:
        return "simplify_error"
    if payload.get("target_checker") != "passed":
        return "checker_error"
    if payload.get("runtime", "passed") != "passed":
        return "runtime_error"
    comparison = payload.get("comparison")
    if comparison and not comparison.get("same_exact"):
        return "mismatch"
    return "exact" if comparison else "valid_target"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int)
    parser.add_argument("--mode", choices=("default", "disabled"))
    args = parser.parse_args()
    if args.worker is not None:
        return worker(args.worker, args.mode)

    rows, failures = [], []
    for index, case in enumerate(CASES):
        for mode in ("default", "disabled"):
            process = subprocess.run([sys.executable, __file__, "--worker", str(index), "--mode", mode], capture_output=True, text=True, timeout=180)
            try:
                payload = json.loads(process.stdout.strip().splitlines()[-1]) if process.stdout.strip() else {}
            except Exception:
                payload = {}
            payload.setdefault("case", case["name"])
            payload.setdefault("mode", mode)
            payload.setdefault("pass", case["pass"])
            actual = classify(process, payload)
            expected = case[f"expected_{mode}"]
            row = {**payload, "actual": actual, "expected": expected, "returncode": process.returncode, "stderr": process.stderr[-2000:]}
            rows.append(row)
            if actual != expected:
                failures.append({"case": case["name"], "mode": mode, "expected": expected, "actual": actual})
    print(json.dumps({"results": rows, "expectation_failures": failures}, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
