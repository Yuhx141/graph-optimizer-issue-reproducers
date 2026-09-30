## Summary

Training ScaledSumFusion bypasses a non-scale producer on the third input of a chained Add. In the reproducer the third branch is Relu(X), but the fused ScaledSum consumes X directly. For X=-3, the source returns 2 while the optimized model returns -1.

Replacing Relu with Identity is the positive control; it reaches the same three-input fusion and remains equivalent.

## Reproduce

Install a CUDA-enabled ORT Training build and run:

    python reproduce.py

The script disables CPU EP fallback. Expected final line:

    REPRODUCED

## Expected result

The fusion should only bypass the third producer when it is a supported scale operator or an Identity. Otherwise it should preserve that branch or skip the fusion.

## Likely cause

When the third producer exists but IsScaleOperator returns false, the fallback branch appends the producer's input rather than the Add input. This silently bypasses Relu and any other unsupported producer:

https://github.com/microsoft/onnxruntime/blob/b2097437af1184d2c0231fca2f2e6468158494b0/orttraining/orttraining/core/optimizer/scaled_sum_fusion.cc#L184-L201

This is separate from #32969, where the same fallback dereferences null when the third Add input comes directly from a graph input.

## Environment

- ORT 1.31.0, revision b2097437af1184d2c0231fca2f2e6468158494b0
- CUDA 12.8.1, RTX 5080 Laptop GPU (sm_120)
- cuDNN 9.8.0.87 system libraries preloaded before CUDA plugin registration
- CPU fallback disabled

## Reproducer

https://github.com/Yuhx141/graph-optimizer-issue-reproducers/tree/main/onnxruntime/ORT-TRAIN-032
