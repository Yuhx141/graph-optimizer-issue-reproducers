## Affected versions

- Latest stable release tested: `v1.30.0` (`f2c39fe2f838cf35ce7da92824f5a5e3ee6e88a7`)
- A local 1.31.0 development CPU wheel reproduces the same behavior.
- The relevant CSE and determinism checks remain present on official `main` `b2097437af1184d2c0231fca2f2e6468158494b0`.

## Summary

Two identical `Bernoulli(X)` nodes represent independent random samples. With BASIC optimization, CommonSubexpressionElimination merges their expanded random chains, so subtracting the two results becomes identically zero.

With optimization disabled, and with only CSE disabled, roughly half of the 1,024 output elements are nonzero on every run. With CSE enabled, all three runs contain zero nonzero elements.

## Steps to reproduce

```bash
git clone https://github.com/Yuhx141/graph-optimizer-issue-reproducers.git
cd graph-optimizer-issue-reproducers/onnxruntime/ORT-SEM-017
python reproduce.py
```

The script runs the source graph with optimization disabled, BASIC optimization, and BASIC with only `CommonSubexpressionElimination` disabled.

## Expected result

CSE should keep independently evaluated nondeterministic operators separate.

## Actual result

CSE leaves one `RandomUniformLike` expansion instead of two, and `Bernoulli(X) - Bernoulli(X)` becomes identically zero.

## Likely cause

`CommonSubexpressionElimination` relies on `optimizer_utils::IsOperationDeterministic`. The ONNX nondeterministic-op list includes Random and Dropout operators, but not `Bernoulli`:

- https://github.com/microsoft/onnxruntime/blob/b2097437af1184d2c0231fca2f2e6468158494b0/onnxruntime/core/optimizer/utils.cc#L275-L293
- https://github.com/microsoft/onnxruntime/blob/b2097437af1184d2c0231fca2f2e6468158494b0/onnxruntime/core/optimizer/common_subexpression_elimination.cc#L358-L366

## Environment

- OS: Ubuntu 24.04.4 LTS, x86-64
- Python: 3.11
- NumPy: 2.4.6
- ONNX: 1.22.0
- ONNX Runtime: 1.30.0 release wheel and a local 1.31.0 development CPU wheel
- Execution provider: CPUExecutionProvider

## Duplicate search

I searched all ONNX Runtime issues and PRs for `Bernoulli`, CSE/CommonSubexpressionElimination, nondeterministic operations, and merged random samples on 2026-09-30. The Bernoulli results concern operator support and ONNX updates; none report this optimizer behavior.

## Reproducer

- https://github.com/Yuhx141/graph-optimizer-issue-reproducers/tree/main/onnxruntime/ORT-SEM-017
