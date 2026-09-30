## Summary

SoftmaxCrossEntropyLossInternalFusion has two missing guards around the LogSoftmax -> NegativeLogLikelihoodLossInternal rewrite:

1. It ignores the LogSoftmax axis attribute. With a [2, 3] input and axis=0, the source loss differs from the fused loss. The axis=1 model is the positive control and remains equal.
2. It removes LogSoftmax even when its output is also a graph output. The resulting model still declares log_prob, but no node produces it; ONNX checking and ORT loading both reject the optimized graph.

Both source models pass ONNX checking and run with CUDA EP while CPU fallback is disabled.

## Reproduce

Install a CUDA-enabled ORT Training build and run:

    python reproduce.py

Expected final line:

    REPRODUCED

## Expected result

The fusion should verify the supported softmax axis and skip removal when the LogSoftmax result is public or has another consumer.

## Likely cause

The transformer recognizes LogSoftmax by op type alone, removes it immediately, and creates a fresh loss output. It does not inspect axis, output edges, or graph.NodeProducesGraphOutput:

https://github.com/microsoft/onnxruntime/blob/b2097437af1184d2c0231fca2f2e6468158494b0/orttraining/orttraining/core/optimizer/loss_rewriter.cc#L17-L45

## Environment

- ORT 1.31.0, revision b2097437af1184d2c0231fca2f2e6468158494b0
- CUDA 12.8.1, RTX 5080 Laptop GPU (sm_120)
- cuDNN 9.8.0.87 system libraries preloaded before CUDA plugin registration
- CPU fallback disabled

## Reproducer

https://github.com/Yuhx141/graph-optimizer-issue-reproducers/tree/main/onnxruntime/ORT-TRAIN-031
