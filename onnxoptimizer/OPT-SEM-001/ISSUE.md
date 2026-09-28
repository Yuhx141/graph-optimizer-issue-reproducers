## Affected versions

- Latest stable release tested: `v0.4.2` (`ed2d224364233dc724ddd94e0081dfd1e5df212a`)
- Default-branch commit tested: `c8d77f3b8326297d7bfb851513079616f7219cdc`

## Summary

Folding Shape followed by Slice uses indices as if Shape started at dimension zero.

## Steps to reproduce

1. Clone the public reproducer repository and enter this case:

   ```bash
   git clone https://github.com/Yuhx141/graph-optimizer-issue-reproducers.git
   cd graph-optimizer-issue-reproducers/onnxoptimizer/OPT-SEM-001
   ```

2. Install the release dependencies listed in [`requirements.txt`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxoptimizer/requirements.txt), or run against the tested source build.
3. Run `python reproduce.py`.

The script validates the source model, runs the implicated pass, disables only that pass as a positive control, and runs a one-condition negative control. The complete readable reproducer is at https://github.com/Yuhx141/graph-optimizer-issue-reproducers/tree/main/onnxoptimizer/OPT-SEM-001.

## Actual result

Forward and reverse witnesses return dimensions from the wrong offset.

## Expected result

The folded constants must refer to the already-sliced Shape output.

## Why the source graph is legal

`source.onnx` passes `onnx.checker.check_model(..., full_check=True)` and strict ONNX shape inference. Its fixed input is in `input.json`, and it executes with ONNX Runtime graph optimizations disabled. Exact hashes are in `SHA256SUMS`.

## Pass attribution

The minimum pass is `eliminate_slice_after_shape`. Disabling only that pass restores a valid graph with output exactly equal to the unoptimized source for this witness.

## One-condition control

`control.onnx` changes the critical condition while keeping the same operator family. It remains valid and exact after optimization; its input is `control-input.json`.

## Likely cause and source location

The pass folds Slice indices against the original input rank without adding Shape.start. See [passes/eliminate_slice_after_shape.h#L1-L111](https://github.com/onnx/optimizer/blob/c8d77f3b8326297d7bfb851513079616f7219cdc/passes/eliminate_slice_after_shape.h#L1-L111).

## Duplicate search

No exact issue or PR found in a directed search performed on 2026-09-28.

## Environment

Linux x86-64, CPython 3.11, ONNX 1.22.0, NumPy 2.4.6, ONNX Runtime 1.30.0, CPU execution provider. Full revision and build details are in [environment.json](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/environment.json).

## Reproducer files

- [`reproduce.py`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxoptimizer/OPT-SEM-001/reproduce.py): direct pass-specific check
- [`source.txt`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxoptimizer/OPT-SEM-001/source.txt): readable ONNX graph
- [`control.txt`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxoptimizer/OPT-SEM-001/control.txt): one-condition control
- Tiny `.onnx` models, JSON inputs, and SHA-256 hashes are in the same directory.

## Previous report

This replaces the withdrawn report [#334](https://github.com/onnx/optimizer/issues/334) with a public, human-readable reproducer.
