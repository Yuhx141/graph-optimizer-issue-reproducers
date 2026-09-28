## Affected versions

- Latest stable release tested: `v1.30.0` (`f2c39fe2f838cf35ce7da92824f5a5e3ee6e88a7`)
- Default-branch commit tested: `a11b4e593164c82f7ce95efce3693744103f8176`

## Summary

Explicit zero Pad is fused into MaxPool implicit padding, whose out-of-bounds values do not behave as zeros.

## Steps to reproduce

1. Clone the public reproducer repository and enter this case:

   ```bash
   git clone https://github.com/Yuhx141/graph-optimizer-issue-reproducers.git
   cd graph-optimizer-issue-reproducers/onnxruntime/ORT-SEM-001
   ```

2. Install the release dependencies listed in [`requirements.txt`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxruntime/requirements.txt), or run against the tested source build.
3. Run `python reproduce.py`.

The script validates the source model, runs the implicated pass, disables only that pass as a positive control, and runs a one-condition negative control. The complete readable reproducer is at https://github.com/Yuhx141/graph-optimizer-issue-reproducers/tree/main/onnxruntime/ORT-SEM-001.

## Actual result

With Pad_Fusion enabled it is -1; disabling only Pad_Fusion restores 0.

## Expected result

The first pooled value for [-1,-2] with a leading zero pad is 0.

## Why the source graph is legal

`source.onnx` passes `onnx.checker.check_model(..., full_check=True)` and strict ONNX shape inference. Its fixed input is in `input.json`, and it executes with ONNX Runtime graph optimizations disabled. Exact hashes are in `SHA256SUMS`.

## Pass attribution

The minimum pass is `Pad_Fusion`. Disabling only that pass restores a valid graph with output exactly equal to the unoptimized source for this witness.

## One-condition control

`control.onnx` changes the critical condition while keeping the same operator family. It remains valid and exact after optimization; its input is `control-input.json`.

## Likely cause and source location

PadFusion permits zero-valued Pad before MaxPool, although MaxPool implicit padding uses the lowest representable value. See [onnxruntime/core/optimizer/pad_fusion.cc#L67-L160](https://github.com/microsoft/onnxruntime/blob/a11b4e593164c82f7ce95efce3693744103f8176/onnxruntime/core/optimizer/pad_fusion.cc#L67-L160).

## Duplicate search

No exact issue or PR found in a directed search performed on 2026-09-28.

## Environment

Linux x86-64, CPython 3.11, ONNX 1.22.0, NumPy 2.4.6, ONNX Runtime 1.30.0 release wheel or the 1.31.0 wheel built from the stated main commit, CPU execution provider. Full revision and build details are in [environment.json](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/environment.json).

## Reproducer files

- [`reproduce.py`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxruntime/ORT-SEM-001/reproduce.py): direct pass-specific check
- [`source.txt`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxruntime/ORT-SEM-001/source.txt): readable ONNX graph
- [`control.txt`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxruntime/ORT-SEM-001/control.txt): one-condition control
- Tiny `.onnx` models, JSON inputs, and SHA-256 hashes are in the same directory.

## Previous report

This replaces the withdrawn report [#32856](https://github.com/microsoft/onnxruntime/issues/32856) with a public, human-readable reproducer.
