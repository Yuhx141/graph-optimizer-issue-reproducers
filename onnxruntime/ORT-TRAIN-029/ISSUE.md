## Summary

Training QDQFusion discards the DequantizeLinear scale and zero point. A legal pair with Q scale 0.1 and DQ scale 0.2 produces [0, 0.4, 0.4, 2.0] before optimization, but the fused FakeQuant produces [0, 0.2, 0.2, 1.0].

The paired equal-scale model (0.1 on both nodes) is the positive control and remains equal after fusion.

## Reproduce

Install a CUDA-enabled ORT Training build and run:

    python reproduce.py

The script disables CPU EP fallback. Expected final line:

    REPRODUCED

## Expected result

QDQFusion should require the Q and DQ quantization parameters to be equivalent, or preserve both operations.

## Likely cause

FuseQDQNodes constructs FakeQuant using only quantize_node.MutableInputDefs(). The pattern check verifies the Q/DQ topology but does not compare the DQ scale or zero point:

https://github.com/microsoft/onnxruntime/blob/b2097437af1184d2c0231fca2f2e6468158494b0/orttraining/orttraining/core/optimizer/qdq_fusion.cc#L64-L82
https://github.com/microsoft/onnxruntime/blob/b2097437af1184d2c0231fca2f2e6468158494b0/orttraining/orttraining/core/optimizer/qdq_fusion.cc#L101-L149

This differs from #32966, which concerns the quantizer's omitted default zero point and quantization range.

## Environment

- ORT 1.31.0, revision b2097437af1184d2c0231fca2f2e6468158494b0
- CUDA 12.8.1, RTX 5080 Laptop GPU (sm_120)
- cuDNN 9.8.0.87 system libraries preloaded before CUDA plugin registration
- CPU fallback disabled

## Reproducer

https://github.com/Yuhx141/graph-optimizer-issue-reproducers/tree/main/onnxruntime/ORT-TRAIN-029
