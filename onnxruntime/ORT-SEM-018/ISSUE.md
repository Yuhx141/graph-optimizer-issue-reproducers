## Affected versions

- Latest stable release tested: `v1.30.0` (`f2c39fe2f838cf35ce7da92824f5a5e3ee6e88a7`)
- A local 1.31.0 development CPU wheel also reproduces it.
- `slice_elimination.cc` is byte-identical in the tested source checkout and official `main` `b2097437af1184d2c0231fca2f2e6468158494b0`.

## Summary

`EliminateSlice` checks `steps` only inside the branch where the optional `axes` input exists. A legal Slice can omit `axes` while still providing `steps`, so `steps=2` is missed and the non-identity Slice is removed.

The minimal graph applies `Slice(starts=[0], ends=[INT64_MAX], axes omitted, steps=[2])` to `[0,1,2,3,4,5]`.

## Reproduce

```bash
git clone https://github.com/Yuhx141/graph-optimizer-issue-reproducers.git
cd graph-optimizer-issue-reproducers/onnxruntime/ORT-SEM-018
python reproduce.py
```

## Expected result

The Slice remains and the output is `[-0, -2, -4]`.

## Actual result

BASIC optimization removes Slice and returns `[-0, -1, -2, -3, -4, -5]`. Disabling only `EliminateSlice` restores the three-element baseline.

## Likely cause

The steps validation at lines 94-109 is nested under `if (get_input_if_exists(3))`:

https://github.com/microsoft/onnxruntime/blob/b2097437af1184d2c0231fca2f2e6468158494b0/onnxruntime/core/optimizer/slice_elimination.cc#L86-L110

The ONNX Slice schema permits omitting `axes`; in that case axes default to `[0, ..., r-1]`, while a supplied `steps` input still applies.

## Environment

- OS: Ubuntu 24.04.4 LTS, x86-64
- Python: 3.11
- NumPy: 2.4.6
- ONNX: 1.22.0
- ONNX Runtime: 1.30.0 release wheel and a local 1.31.0 development CPU wheel
- Execution provider: CPUExecutionProvider

## Duplicate search

I checked all issues and PRs for `EliminateSlice`, optional axes, omitted axes, and steps. #885 was a 2019 subgraph shape-inference failure fixed by #918. #27638/#27718 concern `ConcatSliceElimination`, not this `EliminateSlice` guard. No exact report was found on 2026-09-30.

## Reproducer

- https://github.com/Yuhx141/graph-optimizer-issue-reproducers/tree/main/onnxruntime/ORT-SEM-018
