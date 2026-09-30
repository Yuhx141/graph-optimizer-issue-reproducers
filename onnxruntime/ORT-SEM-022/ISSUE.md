## Affected versions

- Latest stable release tested: `v1.30.0` (`f2c39fe2f838cf35ce7da92824f5a5e3ee6e88a7`)
- Also reproduced with a local 1.31.0 development CPU wheel.
- The affected implementations are unchanged on official `main` `b2097437af1184d2c0231fca2f2e6468158494b0`.

## Summary

`LayerNormFusion` and `SimplifiedLayerNormFusion` both accept DOUBLE graphs, but neither preserves a DOUBLE epsilon that cannot be represented as float.

- `LayerNormFusion` reads epsilon as double and then casts it to float for the fused attribute.
- `SimplifiedLayerNormFusion` reads only a FLOAT epsilon; a DOUBLE epsilon is replaced with the default `1e-5f`.

The attached models use `epsilon=1e-50`. The first fused result changes from about `±0.4472` to `±1`. The simplified result changes from about `0.8165` to `3.16e-23`.

## Reproduce

```bash
git clone https://github.com/Yuhx141/graph-optimizer-issue-reproducers.git
cd graph-optimizer-issue-reproducers/onnxruntime/ORT-SEM-022
python reproduce.py
```

The optimized runs disable `NoopElimination` to isolate this result from #32960. Each positive control additionally disables only the relevant LayerNorm fusion and recovers the baseline exactly.

## Expected result

The optimizer should preserve the DOUBLE epsilon, or skip the fusion when the fused float attribute cannot represent it.

## Actual result

Both source models run with CPUExecutionProvider. Enabling the relevant fusion changes the output deterministically; disabling that fusion restores the unoptimized output exactly.

## Likely cause

Both fusions list DOUBLE as supported and set `stash_type` to DOUBLE, but their epsilon handling loses the source value:

https://github.com/microsoft/onnxruntime/blob/b2097437af1184d2c0231fca2f2e6468158494b0/onnxruntime/core/optimizer/layer_norm_fusion.cc#L16-L29

https://github.com/microsoft/onnxruntime/blob/b2097437af1184d2c0231fca2f2e6468158494b0/onnxruntime/core/optimizer/layer_norm_fusion.cc#L572-L587

https://github.com/microsoft/onnxruntime/blob/b2097437af1184d2c0231fca2f2e6468158494b0/onnxruntime/core/optimizer/layer_norm_fusion.cc#L822-L840

## Environment

- OS: Ubuntu 24.04.4 LTS, x86-64
- Python: 3.11
- NumPy: 2.4.6
- ONNX: 1.22.0
- ONNX Runtime: 1.30.0 release wheel and a local 1.31.0 development CPU wheel
- Execution provider: CPUExecutionProvider

## Duplicate search

I searched all issues and PRs for `LayerNormFusion`, `SimplifiedLayerNormFusion`, `LayerNormalization`, DOUBLE epsilon, epsilon precision/underflow, `stash_type`, and float casts. No issue or PR with the same condition and root cause was found on 2026-09-30.

## Reproducer

- https://github.com/Yuhx141/graph-optimizer-issue-reproducers/tree/main/onnxruntime/ORT-SEM-022
