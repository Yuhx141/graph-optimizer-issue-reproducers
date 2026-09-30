## Summary

The training `QDQFusion` converts a `QuantizeLinear -> DequantizeLinear` pair with an omitted `zero_point` into `FakeQuant(quant_min=-128, quant_max=127)`. ONNX defines the omitted zero point as UINT8 zero, so the equivalent range is `[0, 255]`.

This changes valid negative inputs and sufficiently large positive inputs. With scale `0.1`, the source QDQ pair returns:

```text
[0, 0, 0, 1, 20]
```

The fused `FakeQuant` returns:

```text
[-12.8, -1, 0, 1, 12.7]
```

An otherwise identical model with an explicit UINT8 zero point is a positive control: the fusion emits `quant_min=0, quant_max=255` and preserves the source result.

## Reproduce

```bash
git clone https://github.com/Yuhx141/graph-optimizer-issue-reproducers.git
cd graph-optimizer-issue-reproducers/onnxruntime/ORT-TRAIN-024
python -m pip install onnx onnxruntime-training==1.19.2
python reproduce.py
```

Expected final line:

```text
REPRODUCED
```

## Expected result

When `zero_point` is omitted, `QDQFusion` should use the ONNX default UINT8 type and emit `quant_min=0, quant_max=255`.

## Likely cause

`ReplaceOrCreateZeroPointInitializer` initializes `zero_point_type` to INT8. The no-input branch creates a float zero point but returns that unchanged INT8 type:

https://github.com/microsoft/onnxruntime/blob/b2097437af1184d2c0231fca2f2e6468158494b0/orttraining/orttraining/core/optimizer/qdq_fusion.cc#L15-L66

The current QDQFusion test suite includes a no-zero-point model, but its assertion only checks that Q/DQ became one FakeQuant node; it does not compare the quantization range or output values:

https://github.com/microsoft/onnxruntime/blob/b2097437af1184d2c0231fca2f2e6468158494b0/orttraining/orttraining/test/optimizer/graph_transform_test.cc#L1458-L1482

I searched the repository issues and PRs for `QDQFusion`, `FakeQuant`, omitted/default zero point, and UINT8. I found the original QAT implementation PRs and the per-channel limitation report, but no report of this default-type mismatch.

## Affected versions

- Dynamically reproduced with the latest published `onnxruntime-training` wheel, 1.19.2 (`ffceed9d44`).
- The affected branch is still present in official `main` at `b2097437af1184d2c0231fca2f2e6468158494b0`.

## Environment

- Ubuntu 24.04, x86-64
- Python 3.11
- ONNX 1.22.0
- onnxruntime-training 1.19.2
- CPUExecutionProvider

## Reproducer

- https://github.com/Yuhx141/graph-optimizer-issue-reproducers/tree/main/onnxruntime/ORT-TRAIN-024
