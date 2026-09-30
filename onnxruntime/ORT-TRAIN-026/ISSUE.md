## Summary

The training `TransposeReplacement` assumes that every Transpose node has an explicit `perm` attribute. ONNX makes `perm` optional and defines the default as reversing the input dimensions.

A valid float Transpose with omitted `perm` runs normally before training optimization, but `get_optimized_model` fails with:

```text
IndexError: _Map_base::at
```

An otherwise identical model with the equivalent explicit `perm=[1, 0]` is a positive control: it optimizes and produces the same output.

## Reproduce

```bash
git clone https://github.com/Yuhx141/graph-optimizer-issue-reproducers.git
cd graph-optimizer-issue-reproducers/onnxruntime/ORT-TRAIN-026
python -m pip install onnx onnxruntime-training==1.19.2
python reproduce.py
```

Expected final line:

```text
REPRODUCED
```

## Expected result

`TransposeReplacement` should derive the default reversed permutation when `perm` is absent, or skip the replacement. Omission and an explicitly written default permutation should behave the same.

## Likely cause

The rule has no attribute guard and reads `perm` with `GetAttributes().at("perm")`:

https://github.com/microsoft/onnxruntime/blob/b2097437af1184d2c0231fca2f2e6468158494b0/orttraining/orttraining/core/optimizer/transpose_replacement..cc#L15-L68

I searched the repository issues and PRs for `TransposeReplacement`, omitted/default perm, training Transpose, and `_Map_base::at`. I found no matching report.

## Affected versions

- Dynamically reproduced with the latest published `onnxruntime-training` wheel, 1.19.2 (`ffceed9d44`).
- The unconditional attribute access is still present in official `main` at `b2097437af1184d2c0231fca2f2e6468158494b0`.

## Environment

- Ubuntu 24.04, x86-64
- Python 3.11
- ONNX 1.22.0
- onnxruntime-training 1.19.2
- CPUExecutionProvider

## Reproducer

- https://github.com/Yuhx141/graph-optimizer-issue-reproducers/tree/main/onnxruntime/ORT-TRAIN-026
