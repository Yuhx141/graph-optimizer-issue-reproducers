## Affected versions

- Latest stable release tested: `v1.30.0` (`f2c39fe2f838cf35ce7da92824f5a5e3ee6e88a7`)
- A local 1.31.0 development CPU wheel reproduces the same result.
- `noop_elimination.cc` is unchanged on official `main` `b2097437af1184d2c0231fca2f2e6468158494b0`.

## Summary

NoopElimination converts DOUBLE scalar initializers to `float` before deciding whether Add/Sub/Mul/Div is an identity. The nonzero DOUBLE value `1e-50` underflows to float zero, so an Add node is removed.

## Reproduce

```bash
git clone https://github.com/Yuhx141/graph-optimizer-issue-reproducers.git
cd graph-optimizer-issue-reproducers/onnxruntime/ORT-SEM-020
python reproduce.py
```

## Expected result

The DOUBLE Add remains. For the fixed input, the result is `[-1e-50, -2e-50, -0.0]`.

## Actual result

BASIC optimization removes Add and returns `[0.0, -1e-50, 1e-50]`. Disabling only `NoopElimination` restores the baseline bit for bit.

## Likely cause

Every supported initializer type is stored in a local `float`; the DOUBLE branch explicitly casts to float:

https://github.com/microsoft/onnxruntime/blob/b2097437af1184d2c0231fca2f2e6468158494b0/onnxruntime/core/optimizer/noop_elimination.cc#L74-L103

## Environment

- OS: Ubuntu 24.04.4 LTS, x86-64
- Python: 3.11
- NumPy: 2.4.6
- ONNX: 1.22.0
- ONNX Runtime: 1.30.0 release wheel and a local 1.31.0 development CPU wheel
- Execution provider: CPUExecutionProvider

## Duplicate search

I searched all issues and PRs for `NoopElimination`, DOUBLE/float precision, underflow, and removed identity constants. #32413/#32414/#32416 are invalid-graph pass interactions, and #13460 added more no-op patterns; none covers dtype narrowing. No exact duplicate was found on 2026-09-30.

## Reproducer

- https://github.com/Yuhx141/graph-optimizer-issue-reproducers/tree/main/onnxruntime/ORT-SEM-020
