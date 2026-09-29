## Crash at a legal Shape boundary

ONNX defines `Shape.start` using Python-style slicing and clamps an out-of-range positive start to the input rank. The reduced legal model therefore returns an empty int64 shape vector. Default simplification instead terminates the Python process with SIGSEGV.

The important boundary is `start > rank`; the ONNX definition clamps it just like a Python slice.

## Reproducer

The model is 76 bytes and does not need an input tensor.

```bash
git clone https://github.com/Yuhx141/graph-optimizer-issue-reproducers.git
cd graph-optimizer-issue-reproducers/onnxsim/SIM-CRASH-001
python reproduce.py
```

- `Shape start beyond rank` — skip control: `eliminate_shape_op`

The script checks the source, runs the default simplifier in a child process, and repeats the case with only the named pass skipped. Full files and the recorded run are in [SIM-CRASH-001](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/tree/main/onnxsim/SIM-CRASH-001).

## Observed and expected

Observed: The source passes the ONNX checker. Running default simplification crashes the child process; skipping only `eliminate_shape_op` completes successfully.

Expected: The simplifier should fold the Shape to an empty int64 initializer, or leave the legal Shape unchanged.

Skipping only `eliminate_shape_op` for the corresponding witness removes the failure or mismatch.

## Suspected iterator bug

`patternMatchPredicate` and `runTransform` form iterators with `begin() + start` without first clamping `start` and `end` to the known rank. See [`eliminate_shape_op.h` lines 26-49](https://github.com/onnxsim/optimizer/blob/8c13b168c37e74b610d0cccba4e9e01618c612a7/onnxoptimizer/passes/eliminate_shape_op.h#L26-L49). The [ONNX Shape schema](https://onnx.ai/onnx/operators/onnx__Shape.html) defines the attributes using Python-style slicing.

## Existing reports checked

The full issue/PR history of `onnxsim/onnxsim`, `onnxsim/optimizer`, and `onnx/optimizer` was searched on 2026-09-29.

- No report was found for the `Shape.start > rank` segfault condition.
- [onnx/optimizer#279](https://github.com/onnx/optimizer/issues/279) and [#262](https://github.com/onnx/optimizer/issues/262) report numerical mismatches from the same pass on larger generated models, without this minimized boundary condition or a native crash. They are related but do not establish the same root cause.

## Tested revision

- onnxsim master: `cc035f51c7ee7b8363f8990d19c33562d827bfc4`
- onnxsim/optimizer submodule: `8c13b168c37e74b610d0cccba4e9e01618c612a7`
- Latest release was checked separately where the affected pass exists; the attached result is authoritative for the master revision above.

The [reproducer directory](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/tree/main/onnxsim/SIM-CRASH-001) contains:

- [`reproduce.py`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxsim/SIM-CRASH-001/reproduce.py)
- [`cases.json`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxsim/SIM-CRASH-001/cases.json)
- readable ONNX text and the exact small `.onnx` model(s)
- [`observed-current.json`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxsim/SIM-CRASH-001/observed-current.json)
- [`SHA256SUMS`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxsim/SIM-CRASH-001/SHA256SUMS)

- Ubuntu 24.04.4 LTS, x86_64
- CPython 3.11.16
- ONNX 1.22.0
- ONNX Runtime 1.30.0, CPUExecutionProvider, graph optimization disabled for differential execution
- NumPy 2.4.6
