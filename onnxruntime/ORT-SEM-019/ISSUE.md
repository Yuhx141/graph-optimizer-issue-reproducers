## Affected versions

- Latest stable release tested: `v1.30.0` (`f2c39fe2f838cf35ce7da92824f5a5e3ee6e88a7`)
- Reproduced with a local 1.31.0 development CPU wheel.
- The affected file is unchanged on official `main` `b2097437af1184d2c0231fca2f2e6468158494b0`.

## Summary

`ExpandElimination` treats a target dimension of zero as if it cannot change the input. For an input with shape `[1]`, `Expand` to `[0]` produces a legal empty tensor, so deleting it changes both shape and values.

## Reproduce

```bash
git clone https://github.com/Yuhx141/graph-optimizer-issue-reproducers.git
cd graph-optimizer-issue-reproducers/onnxruntime/ORT-SEM-019
python reproduce.py
```

## Expected result

The output has shape `[0]` and contains no elements.

## Actual result

BASIC optimization removes Expand. The output becomes shape `[1]` with value `[-2]`. Disabling only `ExpandElimination` restores the empty tensor.

## Likely cause

The identity test rejects target dimensions greater than one when they differ, but accepts zero:

https://github.com/microsoft/onnxruntime/blob/b2097437af1184d2c0231fca2f2e6468158494b0/onnxruntime/core/optimizer/expand_elimination.cc#L46-L68

## Environment

- OS: Ubuntu 24.04.4 LTS, x86-64
- Python: 3.11
- NumPy: 2.4.6
- ONNX: 1.22.0
- ONNX Runtime: 1.30.0 release wheel and a local 1.31.0 development CPU wheel
- Execution provider: CPUExecutionProvider

## Duplicate search

I searched all issues and PRs for `ExpandElimination`, zero dimensions, zero target shapes, and empty tensors. #3610 introduced the pass, but neither it nor the other matches describes this zero-dimension case. No exact duplicate was found on 2026-09-30.

## Reproducer

- https://github.com/Yuhx141/graph-optimizer-issue-reproducers/tree/main/onnxruntime/ORT-SEM-019
