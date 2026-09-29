## Affected versions

- Latest stable release tested: `v1.30.0` (`f2c39fe2f838cf35ce7da92824f5a5e3ee6e88a7`)
- Default-branch build tested: `a11b4e593164c82f7ce95efce3693744103f8176`

## Summary

the optimized graph has no producer for the public Cast output named cast.

The source model loads with optimizations disabled. Enabling graph optimization reproduces the failure on ONNX Runtime 1.30.0. Disabling only `MatmulTransposeFusion` removes the failure. A compiled 1.31 development wheel based on `a11b4e593164c82f7ce95efce3693744103f8176` reproduced the same root. The relevant source file is byte-identical at official main `96f73115c95968a3f31f2a110b33c164d847dda4`.

## Steps to reproduce

1. Clone the public reproducer and enter this case:

   ```bash
   git clone https://github.com/Yuhx141/graph-optimizer-issue-reproducers.git
   cd graph-optimizer-issue-reproducers/onnxruntime/ORT-SEM-006
   ```

2. Install the versions in [`requirements.txt`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxruntime/requirements.txt).
3. Run `python reproduce.py`.

The standalone script validates the source behavior, reproduces the optimized behavior, and disables the implicated transformer as a positive control. The readable reproducer is at https://github.com/Yuhx141/graph-optimizer-issue-reproducers/tree/main/onnxruntime/ORT-SEM-006.

## Expected result

The public Cast value must retain a producer after Cast/Transpose reordering, or the pass must decline the rewrite.

## Actual result

ORT_ENABLE_ALL fails with “Failed to find node output or a constant initializer producing output: cast.” Disabling only MatmulTransposeFusion recovers.

## Likely cause and source location

ReorderCastAndTranspose reuses the original Cast output name for a new Transpose, then removes nodes using ordinary consumer counts without preserving graph-output ownership.

Relevant source locations:
- `onnxruntime/core/optimizer/matmul_transpose_fusion.cc:137-190`

## Duplicate search

No exact issue or PR was found by pass-name, missing-producer, Cast-output, and public-output searches on 2026-09-29.
- No exact upstream issue or PR found.

## Environment

- OS: Ubuntu 24.04.4 LTS, x86-64
- CPU: Intel Core Ultra 5 225
- Python: 3.11.16
- NumPy: 2.4.6
- ONNX: 1.22.0
- ONNX Runtime: 1.30.0 (`f2c39fe2f838cf35ce7da92824f5a5e3ee6e88a7`)
- Execution provider: CPUExecutionProvider
- Optimization level: ORT_ENABLE_ALL

## Reproducer files

- [`reproduce.py`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxruntime/ORT-SEM-006/reproduce.py): standalone self-check
- [`source.txt`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxruntime/ORT-SEM-006/source.txt): readable ONNX graph
- [`minimal_model.onnx`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxruntime/ORT-SEM-006/minimal_model.onnx): exact source model
- [`SHA256SUMS`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxruntime/ORT-SEM-006/SHA256SUMS): file hashes
