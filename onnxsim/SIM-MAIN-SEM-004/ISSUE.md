## Is this tolerance intentional?

I found two formula matchers that accept nearby constants with `1e-3` absolute tolerance. I am unsure whether these passes promise an approximate rewrite or exact simplification, so I am filing this as a question with concrete witnesses.

The pattern matchers use an absolute tolerance of `1e-3` for constants that define the formula itself. That accepts expressions which are close in parameter space but not equivalent computations.

## Cases

One case perturbs the GELU coefficient; the other perturbs the LayerNorm exponent.

```bash
git clone https://github.com/Yuhx141/graph-optimizer-issue-reproducers.git
cd graph-optimizer-issue-reproducers/onnxsim/SIM-MAIN-SEM-004
python reproduce.py
```

- `GELU coefficient 0.5009` — skip control: `fuse_gelu`
- `LayerNorm exponent 2.0005` — skip control: `fuse_layer_norm`

The script checks the source, runs the default simplifier in a child process, and repeats the case with only the named pass skipped. Full files and the recorded run are in [SIM-MAIN-SEM-004](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/tree/main/onnxsim/SIM-MAIN-SEM-004).

## What changes

A GELU coefficient of `0.5009` is fused to exact GELU and changes output by `0.0359993`. A LayerNorm exponent of `2.0005` produces NaNs for negative centered values in the source, while the fused target is finite. Both pass-disabled controls preserve the source.

Skipping only `fuse_gelu`, `fuse_layer_norm` for the corresponding witness removes the failure or mismatch.

## Expected contract

Formula-defining constants must match the exact representable target formula, or the pass must prove a stated approximation contract instead of silently applying an exact rewrite.

## Matcher locations

Both matchers define `IsScalarConstantApprox(..., tol=1e-3)`: [`fuse_gelu.h` lines 71-74](https://github.com/onnxsim/onnxsim/blob/cc035f51c7ee7b8363f8990d19c33562d827bfc4/onnxsim/passes/fuse_gelu.h#L71-L74) and [`fuse_layer_norm.h` lines 89-92](https://github.com/onnxsim/onnxsim/blob/cc035f51c7ee7b8363f8990d19c33562d827bfc4/onnxsim/passes/fuse_layer_norm.h#L89-L92). The tolerant checks are then used for GELU's constants and LayerNorm's Pow exponent.

If approximate fusion is intended, documenting the error contract would answer the question. Otherwise these constants probably need exact or representation-aware matching.

## Related search

The full issue/PR history of `onnxsim/onnxsim`, `onnxsim/optimizer`, and `onnx/optimizer` was searched on 2026-09-29.

- No exact report was found for the `1e-3` formula-pattern tolerance.
- The fusion PR [onnxsim/onnxsim#656](https://github.com/onnxsim/onnxsim/pull/656) describes the rewrites as precision-preserving, but its tests use the intended constants and do not exercise nearby non-equivalent values.

## Revision and files

- onnxsim master: `cc035f51c7ee7b8363f8990d19c33562d827bfc4`
- onnxsim/optimizer submodule: `8c13b168c37e74b610d0cccba4e9e01618c612a7`
- Latest release was checked separately where the affected pass exists; the attached result is authoritative for the master revision above.

The [reproducer directory](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/tree/main/onnxsim/SIM-MAIN-SEM-004) contains:

- [`reproduce.py`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxsim/SIM-MAIN-SEM-004/reproduce.py)
- [`cases.json`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxsim/SIM-MAIN-SEM-004/cases.json)
- readable ONNX text and the exact small `.onnx` model(s)
- [`observed-current.json`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxsim/SIM-MAIN-SEM-004/observed-current.json)
- [`SHA256SUMS`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxsim/SIM-MAIN-SEM-004/SHA256SUMS)

- Ubuntu 24.04.4 LTS, x86_64
- CPython 3.11.16
- ONNX 1.22.0
- ONNX Runtime 1.30.0, CPUExecutionProvider, graph optimization disabled for differential execution
- NumPy 2.4.6
