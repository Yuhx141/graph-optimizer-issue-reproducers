## Summary

Training CastSceLossFusion dereferences a null producer when SoftmaxCrossEntropyLossInternal receives its scores directly from a graph input. The source model passes ONNX checking and runs on CUDA, but get_optimized_model terminates the subprocess with SIGSEGV.

An equivalent model with Identity(scores) is the positive control: optimization finishes, and source and optimized outputs match.

## Reproduce

Install a CUDA-enabled ORT Training build and run:

    python reproduce.py

Expected final line:

    REPRODUCED

## Expected result

If the scores input has no producer, the transformer should skip the Cast pattern.

## Likely cause

The transformer obtains the producer and immediately dereferences it in IsSupportedOptypeVersionAndDomain without a null check:

https://github.com/microsoft/onnxruntime/blob/b2097437af1184d2c0231fca2f2e6468158494b0/orttraining/orttraining/core/optimizer/cast_sce_loss_fusion.cc#L31-L34

## Environment

- ORT 1.31.0, revision b2097437af1184d2c0231fca2f2e6468158494b0
- CUDA 12.8.1, RTX 5080 Laptop GPU (sm_120)
- cuDNN 9.8.0.87 system libraries preloaded before CUDA plugin registration
- CPU fallback disabled for runtime checks

## Reproducer

https://github.com/Yuhx141/graph-optimizer-issue-reproducers/tree/main/onnxruntime/ORT-TRAIN-030
