## Affected versions

- Latest stable release tested: `v1.30.0` (`f2c39fe2f838cf35ce7da92824f5a5e3ee6e88a7`)
- Also reproduced with a local 1.31.0 development CPU wheel.
- The affected implementation is unchanged on official `main` `b2097437af1184d2c0231fca2f2e6468158494b0`.

## Summary

MatMulScaleFusion accepts DOUBLE MatMul after #28145, but reads every scalar scale into a C++ `float` and stores the fused `alpha` as a float attribute. A representable DOUBLE scale can therefore change during fusion.

The reproducer uses `nextafter(1.0, 2.0)` as the DOUBLE multiplier. It is distinct from 1 in DOUBLE and rounds to exactly 1 in float.

## Reproduce

```bash
git clone https://github.com/Yuhx141/graph-optimizer-issue-reproducers.git
cd graph-optimizer-issue-reproducers/onnxruntime/ORT-SEM-021
python reproduce.py
```

The optimized run disables `NoopElimination` so this result is isolated from the separate DOUBLE identity issue. The positive control additionally disables `MatMulScaleFusion`.

## Expected result

The optimizer should preserve the DOUBLE multiplier, or skip fusion when `alpha` cannot represent it.

## Actual result

The source/control outputs contain values such as `-19.000000000000004`. After fusion they become `-19.0`, `-22.0`, `-43.0`, and `-50.0`.

## Likely cause

`ExtractScalarAsFloatDispatchTarget<double>` casts the scalar to float, `ScaleMergeInfo` stores float, and the final `alpha` is a float attribute:

https://github.com/microsoft/onnxruntime/blob/b2097437af1184d2c0231fca2f2e6468158494b0/onnxruntime/core/optimizer/matmul_scale_fusion.cc#L18-L25
https://github.com/microsoft/onnxruntime/blob/b2097437af1184d2c0231fca2f2e6468158494b0/onnxruntime/core/optimizer/matmul_scale_fusion.cc#L44-L74
https://github.com/microsoft/onnxruntime/blob/b2097437af1184d2c0231fca2f2e6468158494b0/onnxruntime/core/optimizer/matmul_scale_fusion.cc#L234-L248

## Environment

- OS: Ubuntu 24.04.4 LTS, x86-64
- Python: 3.11
- NumPy: 2.4.6
- ONNX: 1.22.0
- ONNX Runtime: 1.30.0 release wheel and a local 1.31.0 development CPU wheel
- Execution provider: CPUExecutionProvider

## Duplicate search

I searched all issues and PRs for `MatMulScaleFusion`, `FusedMatMul`, DOUBLE alpha, and scale precision. #24407/#24492 concern leading-dimension broadcasting. #28145 intentionally added DOUBLE support but tests an exactly representable scale and does not report this precision loss. No exact duplicate was found on 2026-09-30.

## Reproducer

- https://github.com/Yuhx141/graph-optimizer-issue-reproducers/tree/main/onnxruntime/ORT-SEM-021
