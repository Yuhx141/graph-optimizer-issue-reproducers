"""Reproduce optimization/serialization contract failures and paired controls."""
from pathlib import Path
import copy
import json
import sys
import argparse
import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path, nargs="?", default=ROOT / ("run-" + ort.__version__))
CANDIDATES = ["int32-dq-conv", "empty-batch-softmax", "ir3-created-initializer",
              "cast-name-collision", "small-scale-softmax", "per-axis-averagepool"]
parser.add_argument("--candidate", choices=CANDIDATES)
args = parser.parse_args()
OUT = args.output
OUT.mkdir(parents=True, exist_ok=True)


def run(model, feeds, level, path=None, disabled=()):
    options = ort.SessionOptions()
    options.graph_optimization_level = level
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    if path:
        options.optimized_model_filepath = str(path)
    session = ort.InferenceSession(model.SerializeToString(), options,
                                  providers=["CPUExecutionProvider"],
                                  disabled_optimizers=set(disabled))
    return session.run(None, feeds)


def equal(left, right):
    return len(left) == len(right) and all(
        a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)
        for a, b in zip(left, right)
    )


rows = []
for name in CANDIDATES:
    if args.candidate and name != args.candidate:
        continue
    directory = ROOT / name
    if name == "cast-name-collision":
        for variant in ("local", "scope"):
            d = directory / variant
            model = onnx.load(d / "source.onnx")
            onnx.checker.check_model(model, full_check=True)
            with np.load(d / "input.npz", allow_pickle=False) as inputs:
                feeds = dict(inputs)
            source = run(model, feeds, ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
            row = {"case": name + "/" + variant, "source_checker": "passed", "source_runtime": "passed"}
            path = OUT / (name + "-" + variant + "-optimized.onnx")
            try:
                run(model, feeds, ort.GraphOptimizationLevel.ORT_ENABLE_ALL, path)
                onnx.checker.check_model(onnx.load(path), full_check=True)
            except (ort.capi.onnxruntime_pybind11_state.Fail,
                    ort.capi.onnxruntime_pybind11_state.InvalidGraph,
                    onnx.checker.ValidationError) as error:
                row["optimized_error"] = str(error)
            assert "optimized_error" in row and "InsertedPrecisionFreeCast" in row["optimized_error"], row
            assert ("Type Error" if variant == "local" else "single static assignment") in row["optimized_error"], row
            control = onnx.load(d / "control.onnx")
            onnx.checker.check_model(control, full_check=True)
            assert [x.name for x in model.graph.output] == [x.name for x in control.graph.output]
            row["saved_name_only_control_exact"] = equal(source, run(control, feeds, ort.GraphOptimizationLevel.ORT_DISABLE_ALL))
            assert row["saved_name_only_control_exact"], row
            if (d / "control-source.onnx").exists():
                control_source = onnx.load(d / "control-source.onnx")
                onnx.checker.check_model(control_source, full_check=True)
                control_path = OUT / (name + "-local-control-optimized.onnx")
                row["source_rename_control_exact"] = equal(source, run(control_source, feeds, ort.GraphOptimizationLevel.ORT_ENABLE_ALL, control_path))
                onnx.checker.check_model(onnx.load(control_path), full_check=True)
                assert row["source_rename_control_exact"], row
            rows.append(row)
        continue
    if name == "per-axis-averagepool":
        for version in (13, 18):
            model = onnx.load(directory / "source.onnx")
            next(x for x in model.opset_import if not x.domain).version = version
            onnx.checker.check_model(model, full_check=True)
            with np.load(directory / "input.npz", allow_pickle=False) as inputs:
                feeds = dict(inputs)
            source = run(model, feeds, ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
            row = {"case": name + "/opset" + str(version), "source_checker": "passed", "source_runtime": "passed"}
            try:
                run(model, feeds, ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
            except ort.capi.onnxruntime_pybind11_state.Fail as error:
                row["optimized_error"] = str(error)
            assert "QLinearAveragePool" in row.get("optimized_error", "") and "must be a scalar" in row["optimized_error"], row
            row["disabled_pass_exact"] = equal(source, run(model, feeds, ort.GraphOptimizationLevel.ORT_ENABLE_ALL, disabled=("QDQSelectorActionTransformer",)))
            assert row["disabled_pass_exact"], row
            scalar = copy.deepcopy(model)
            for i, initializer in enumerate(scalar.graph.initializer):
                value = onnx.numpy_helper.to_array(initializer)
                if value.size == 1:
                    scalar.graph.initializer[i].CopyFrom(onnx.numpy_helper.from_array(value.reshape(()), initializer.name))
            onnx.checker.check_model(scalar, full_check=True)
            row["scalar_control_exact"] = equal(source, run(scalar, feeds, ort.GraphOptimizationLevel.ORT_ENABLE_ALL))
            assert row["scalar_control_exact"], row
            rows.append(row)
        continue
    if name == "small-scale-softmax":
        model = onnx.load(directory / "source.onnx")
        onnx.checker.check_model(model, full_check=True)
        with np.load(directory / "input.npz", allow_pickle=False) as inputs:
            feeds = dict(inputs)
        source = run(model, feeds, ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
        target = run(model, feeds, ort.GraphOptimizationLevel.ORT_ENABLE_ALL, OUT / (name + "-optimized.onnx"))
        assert all(np.all(x == np.float32(255 / 256)) for x in source), source
        assert all(np.all(x == 0) for x in target), target
        row = {"case": name, "source_checker": "passed", "source_runtime": "passed", "source_output": [x.tolist() for x in source], "optimized_output": [x.tolist() for x in target]}
        row["disabled_pass_exact"] = equal(source, run(model, feeds, ort.GraphOptimizationLevel.ORT_ENABLE_ALL, disabled=("QDQSelectorActionTransformer",)))
        assert row["disabled_pass_exact"], row
        control = onnx.load(directory / "standard-scale-source.onnx")
        onnx.checker.check_model(control, full_check=True)
        row["standard_scale_control_exact"] = equal(run(control, feeds, ort.GraphOptimizationLevel.ORT_DISABLE_ALL), run(control, feeds, ort.GraphOptimizationLevel.ORT_ENABLE_ALL))
        assert row["standard_scale_control_exact"], row
        rows.append(row)
        continue
    model = onnx.load(directory / "source.onnx")
    onnx.checker.check_model(model, full_check=True)
    with np.load(directory / "input.npz", allow_pickle=False) as inputs:
        feeds = dict(inputs)
    source = run(model, feeds, ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
    row = {"case": name, "source_checker": "passed", "source_runtime": "passed",
           "source_outputs": [{"shape": list(x.shape), "dtype": str(x.dtype)} for x in source]}
    target_path = OUT / (name + "-optimized.onnx")
    try:
        target_values = run(model, feeds, ort.GraphOptimizationLevel.ORT_ENABLE_ALL, target_path)
        row["optimized_in_memory"] = "passed"
        row["in_memory_exact"] = equal(source, target_values)
    except Exception as error:
        row["optimized_error"] = type(error).__name__ + ": " + str(error)
    if name != "ir3-created-initializer":
        assert "optimized_error" in row, row
        guarded = run(model, feeds, ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
                      disabled=("QDQSelectorActionTransformer",))
        row["disabled_pass_exact"] = equal(source, guarded)
        assert row["disabled_pass_exact"], row
        control = copy.deepcopy(model)
        control_feeds = feeds
        if name == "int32-dq-conv":
            for node in control.graph.node:
                if node.op_type == "Cast":
                    for attribute in node.attribute:
                        if attribute.name == "to":
                            attribute.i = onnx.TensorProto.INT8
                    for value in control.graph.value_info:
                        if value.name in node.output:
                            value.type.tensor_type.elem_type = onnx.TensorProto.INT8
        else:
            for value in [*control.graph.input, *control.graph.output, *control.graph.value_info]:
                dims = value.type.tensor_type.shape.dim
                if dims and dims[0].HasField("dim_value") and dims[0].dim_value == 0:
                    dims[0].dim_value = 1
            control_feeds = {k: np.zeros((1, *v.shape[1:]), dtype=v.dtype) for k, v in feeds.items()}
        onnx.checker.check_model(control, full_check=True)
        control_source = run(control, control_feeds, ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
        control_target = run(control, control_feeds, ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
        row["neighbor_control_runtime"] = "passed"
        row["neighbor_control_exact"] = equal(control_source, control_target)
    else:
        assert row.get("optimized_in_memory") == "passed", row
        target = onnx.load(target_path)
        try:
            onnx.checker.check_model(target, full_check=True)
        except onnx.checker.ValidationError as error:
            row["target_checker_error"] = str(error)
        assert "target_checker_error" in row, row
        names = {v.name for v in target.graph.input}
        missing = [v for v in target.graph.initializer if v.name not in names]
        row["missing_initializer_input_declarations"] = [v.name for v in missing]
        assert target.ir_version == 3 and missing, row
        fixed = copy.deepcopy(target)
        for initializer in missing:
            fixed.graph.input.append(onnx.helper.make_tensor_value_info(
                initializer.name, initializer.data_type, initializer.dims))
        onnx.checker.check_model(fixed, full_check=True)
        row["declaration_control_exact"] = equal(
            source, run(fixed, feeds, ort.GraphOptimizationLevel.ORT_DISABLE_ALL))
        assert row["declaration_control_exact"], row
        ir4 = copy.deepcopy(model)
        ir4.ir_version = 4
        ir4.graph.ClearField("input")
        onnx.checker.check_model(ir4, full_check=True)
        ir4_path = OUT / "ir4-control-optimized.onnx"
        ir4_value = run(ir4, feeds, ort.GraphOptimizationLevel.ORT_ENABLE_ALL, ir4_path)
        onnx.checker.check_model(onnx.load(ir4_path), full_check=True)
        row["ir4_control_checker"] = "passed"
        row["ir4_control_exact"] = equal(source, ir4_value)
        assert row["ir4_control_exact"], row
    rows.append(row)
expected_cases = 8 if not args.candidate else (2 if args.candidate in ("cast-name-collision", "per-axis-averagepool") else 1)
assert len(rows) == expected_cases
result = {"ort": ort.__version__, "build": ort.get_build_info(), "onnx": onnx.__version__,
          "numpy": np.__version__, "python": sys.version, "cases": rows}
(OUT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
