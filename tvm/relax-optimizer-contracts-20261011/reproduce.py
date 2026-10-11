"""Validate native Relax optimization candidates with executable neighboring controls."""
from pathlib import Path
import json
import argparse
import sys
import numpy as np
import tvm
from tvm import relax
from tvm.script import ir as I
from tvm.script import relax as R

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path, nargs="?", default=ROOT / "fresh-run")
parser.add_argument("--candidate", choices=["002", "005", "006", "017", "060"])
args = parser.parse_args()
SELECT = args.candidate
OUT = args.output
OUT.mkdir(parents=True, exist_ok=True)


def parse(code):
    code = code[code.index("@I.ir_module"):]
    return tvm.script.from_source(code, extra_vars={"I": I, "R": R})


def execute(mod, inputs):
    executable = relax.build(mod, target="llvm", relax_pipeline="default")
    vm = relax.VirtualMachine(executable, tvm.cpu())
    return vm["main"](*[tvm.runtime.tensor(x) for x in inputs]).numpy()


def inputs_for(mod):
    result = []
    for param in mod["main"].params:
        ty = param.ty
        shape = tuple(int(x) for x in ty.shape)
        values = np.arange(1, np.prod(shape) + 1, dtype=np.float32).reshape(shape) / 10
        result.append(values.astype(str(ty.dtype)))
    return result


def check(number, pass_name, risk, control, risk_inputs=None, control_inputs=None, clean_risk=False):
    directory = OUT / ("CAND-" + number)
    directory.mkdir(exist_ok=True)
    (directory / "source.py").write_text(risk.script(show_meta=True))
    (directory / "control.py").write_text(control.script(show_meta=True))
    assert relax.analysis.check_well_formed(risk, check_ty=True)
    assert relax.analysis.check_well_formed(control, check_ty=True)
    risk_inputs = inputs_for(risk) if risk_inputs is None else risk_inputs
    control_inputs = inputs_for(control) if control_inputs is None else control_inputs
    executable_risk = relax.transform.DeadCodeElimination()(risk) if clean_risk else risk
    source = execute(executable_risk, risk_inputs)
    row = {"candidate": "CAND-" + number, "pass": pass_name,
           "source_well_formed": True, "source_executed": True,
           "source_execution_dce_for_unused_pure_binding": clean_risk,
           "source_output_shape": list(source.shape)}
    transform = getattr(relax.transform, pass_name)
    try:
        target = transform()(risk)
        well_formed = relax.analysis.check_well_formed(target, check_ty=True)
        row["target_well_formed"] = bool(well_formed)
        (directory / "target.py").write_text(target.script(show_meta=True))
        if well_formed:
            value = execute(target, risk_inputs)
            row["target_exact"] = bool(np.array_equal(source, value))
        else:
            row["failure"] = "target_not_well_formed"
    except Exception as error:
        row["failure"] = "pass_or_target_error"
        row["error"] = type(error).__name__ + ": " + str(error)
    assert "failure" in row or row.get("target_exact") is False, row
    before = execute(control, control_inputs)
    after_mod = transform()(control)
    assert relax.analysis.check_well_formed(after_mod, check_ty=True)
    after = execute(after_mod, control_inputs)
    np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-6)
    row["control_executed"] = True
    row["control_match"] = True
    row["control_exact"] = bool(np.array_equal(before, after))
    row["control_max_abs"] = float(np.max(np.abs(before - after))) if before.size else 0.0
    (directory / "result.json").write_text(json.dumps(row, indent=2) + "\n")
    print(number, row.get("failure"), "control match", flush=True)
    return row


rows = []
code = '''@I.ir_module
class Before:
    @R.function
    def main(a: R.Tensor([100, 3], "float16"), b: R.Tensor([3, 4], "float16"), c: R.Tensor([4, 2], "float32")):
        inner = R.matmul(a, b, out_dtype="float32")
        return R.matmul(inner, c, out_dtype="float32")
'''
if SELECT in (None, "002"):
    rows.append(check("002", "AdjustMatmulOrder", parse(code), parse(code.replace('"float16"', '"float32"'))))
for number, pass_name in [("005", "ExpandTupleArguments"), ("006", "RemoveUnusedParameters")]:
    if SELECT not in (None, number):
        continue
    code = (ROOT / ("CAND-" + number) / "module.py").read_text()
    if number == "005":
        control_code = code.replace('        closure = R.make_closure(Before.pair_add, ())\n', '')
        start = control_code.index('        return R.invoke_pure_closure(')
        control_code = control_code[:start] + '        return Before.pair_add((x, y))\n'
    else:
        control_code = code.replace('        closure = R.make_closure(Before.lifted, (x, z))\n', '')
        start = control_code.index('        return R.invoke_pure_closure(')
        control_code = control_code[:start] + '        return Before.lifted(y, x, z)\n'
    rows.append(check(number, pass_name, parse(code), parse(control_code)))


def batchnorm(dtype):
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorType([1, 1, 2, 2], dtype))
    weight = relax.const(np.ones((1, 1, 1, 1), dtype=dtype))
    constants = [relax.const(np.array([v], dtype="float32")) for v in [2, 3, 1, 4]]
    with bb.function("main", [x]):
        with bb.dataflow():
            conv = bb.emit(relax.op.nn.conv2d(x, weight, out_dtype="float32"))
            bn = bb.emit(relax.op.nn.batch_norm(conv, *constants, axis=1, training=False))
            out = bb.emit_output(relax.TupleGetItem(bn, 0))
        bb.emit_func_output(out)
    return bb.finalize()


if SELECT in (None, "017"):
    rows.append(check("017", "FoldBatchnormToConv2D", batchnorm("float16"), batchnorm("float32")))
if SELECT in (None, "060"):
    namespace = {}
    code = (ROOT / "CAND-060" / "original-repro.py").read_text().split("for shape in ((2,), (2, 2)):")[0]
    exec(code, namespace)
    rows.append(check("060", "FuseTransposeMatmul", namespace["model"]((2,)), namespace["model"]((2, 2))))
assert len(rows) == (1 if SELECT else 5)
(OUT / "result.json").write_text(json.dumps({"tvm": tvm.__version__, "cases": rows}, indent=2) + "\n")
