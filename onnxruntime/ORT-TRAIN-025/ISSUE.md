## Summary

The training `QDQFusion` does not preserve two kinds of externally used graph values:

1. If the QuantizeLinear result `Q` is both a graph output and consumed by DequantizeLinear, fusion removes Q/DQ and leaves graph output `Q` without a producer. `get_optimized_model` returns a model that fails the ONNX checker and cannot be loaded.
2. If two QDQ pairs share one zero-point initializer, fusing the first pair removes that initializer from the graph. Fusing the second pair then fails because the same initializer is no longer present.

Both source models pass the ONNX checker and run successfully before the training transformation.

## Reproduce

```bash
git clone https://github.com/Yuhx141/graph-optimizer-issue-reproducers.git
cd graph-optimizer-issue-reproducers/onnxruntime/ORT-TRAIN-025
python -m pip install onnx onnxruntime-training==1.19.2
python reproduce.py
```

The public-Q model reports:

```text
Graph output 'Q' is not an output of any node in graph.
```

The shared-initializer model fails inside QDQFusion with:

```text
Expected: zero point initializer with name zero to be present in the graph. Actual: not found.
```

## Expected result

QDQFusion should skip patterns whose Q value is externally visible, and it should keep or clone a zero-point initializer that still has other consumers.

## Likely cause

The pattern check counts node consumers but does not check whether Q is a graph output. The replacement also removes the original zero-point initializer unconditionally:

https://github.com/microsoft/onnxruntime/blob/b2097437af1184d2c0231fca2f2e6468158494b0/orttraining/orttraining/core/optimizer/qdq_fusion.cc#L15-L117

The current tests check only node counts after fusion. They do not cover a public Q value or a zero point shared by two pairs:

https://github.com/microsoft/onnxruntime/blob/b2097437af1184d2c0231fca2f2e6468158494b0/orttraining/orttraining/test/optimizer/graph_transform_test.cc#L1458-L1482

This is related to the graph-visibility invariant in #32965, but it occurs in the separate training-only `QDQFusion` path; the shared-initializer failure is an additional case.

I searched the repository issues and PRs for QDQFusion, FakeQuant, shared zero-point initializers, public QuantizeLinear outputs, and the exact error text. I found no matching report.

## Affected versions

- Dynamically reproduced with the latest published `onnxruntime-training` wheel, 1.19.2 (`ffceed9d44`).
- Both affected operations are still present in official `main` at `b2097437af1184d2c0231fca2f2e6468158494b0`.

## Environment

- Ubuntu 24.04, x86-64
- Python 3.11
- ONNX 1.22.0
- onnxruntime-training 1.19.2
- CPUExecutionProvider

## Reproducer

- https://github.com/Yuhx141/graph-optimizer-issue-reproducers/tree/main/onnxruntime/ORT-TRAIN-025
