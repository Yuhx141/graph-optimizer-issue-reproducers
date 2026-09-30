## Summary

IsInfReduceSumFusion replaces ReduceSum(IsInf(x)) > threshold with Not(IsAllFinite(x)) without checking the Greater threshold. With exactly one infinite value and threshold=1, the source returns false, while the fused graph returns true.

A paired threshold=0 model is the positive control: it fuses to the same IsAllFinite -> Not pattern and both sides return true.

## Reproduce

Install a CUDA-enabled ORT Training build and run:

    python reproduce.py

The script disables CPU EP fallback for both source and optimized sessions. Expected final line:

    REPRODUCED

## Expected result

The fusion should require a scalar zero threshold (and the corresponding Greater input order), or skip the pattern.

## Likely cause

The transformer verifies the IsInf -> Cast -> ReduceSum -> Greater node sequence, but never reads either input of Greater before replacing the whole expression:

https://github.com/microsoft/onnxruntime/blob/b2097437af1184d2c0231fca2f2e6468158494b0/onnxruntime/core/optimizer/isinf_reducesum_fusion.cc#L88-L114

## Environment

- ORT 1.31.0, revision b2097437af1184d2c0231fca2f2e6468158494b0
- CUDA 12.8.1, RTX 5080 Laptop GPU (sm_120)
- cuDNN 9.8.0.87 system libraries preloaded before CUDA plugin registration
- CPU fallback disabled

## Reproducer

https://github.com/Yuhx141/graph-optimizer-issue-reproducers/tree/main/onnxruntime/ORT-TRAIN-028
