## Affected versions

- Latest stable release tested: `v0.4.2` (`ed2d224364233dc724ddd94e0081dfd1e5df212a`)
- Default-branch commit tested: `c8d77f3b8326297d7bfb851513079616f7219cdc`

## Summary

Explicit zero Pad is not equivalent to MaxPool implicit padding for negative inputs.

## Steps to reproduce

1. Clone the public reproducer repository and enter this case:

   ```bash
   git clone https://github.com/Yuhx141/graph-optimizer-issue-reproducers.git
   cd graph-optimizer-issue-reproducers/onnxoptimizer/OPT-SEM-002
   ```

2. Install the release dependencies listed in [`requirements.txt`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxoptimizer/requirements.txt), or run against the tested source build.
3. Run `python reproduce.py`.

The script validates the source model, runs the implicated pass, disables only that pass as a positive control, and runs a one-condition negative control. The complete readable reproducer is at https://github.com/Yuhx141/graph-optimizer-issue-reproducers/tree/main/onnxoptimizer/OPT-SEM-002.

## Actual result

After fusion the first value is -1 because implicit MaxPool padding does not contribute zero.

## Expected result

The first pooled value for [-1,-2] with a leading zero pad is 0.

## Why the source graph is legal

`source.onnx` passes `onnx.checker.check_model(..., full_check=True)` and strict ONNX shape inference. Its fixed input is in `input.json`, and it executes with ONNX Runtime graph optimizations disabled. Exact hashes are in `SHA256SUMS`.

## Pass attribution

The minimum pass is `fuse_pad_into_pool`. Disabling only that pass restores a valid graph with output exactly equal to the unoptimized source for this witness.

## One-condition control

`control.onnx` changes the critical condition while keeping the same operator family. It remains valid and exact after optimization; its input is `control-input.json`.

## Likely cause and source location

The pass copies pad widths but does not preserve the explicit Pad constant value semantics. See [passes/fuse_pad_into_pool.h#L1-L152](https://github.com/onnx/optimizer/blob/c8d77f3b8326297d7bfb851513079616f7219cdc/passes/fuse_pad_into_pool.h#L1-L152).

## Duplicate search

[Issue #269](https://github.com/onnx/optimizer/issues/269) discusses a small Pad/Pool numeric difference, not this exact semantic mismatch.

## Environment

Linux x86-64, CPython 3.11, ONNX 1.22.0, NumPy 2.4.6, ONNX Runtime 1.30.0, CPU execution provider. Full revision and build details are in [environment.json](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/environment.json).

## Reproducer files

- [`reproduce.py`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxoptimizer/OPT-SEM-002/reproduce.py): direct pass-specific check
- [`source.txt`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxoptimizer/OPT-SEM-002/source.txt): readable ONNX graph
- [`control.txt`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxoptimizer/OPT-SEM-002/control.txt): one-condition control
- Tiny `.onnx` models, JSON inputs, and SHA-256 hashes are in the same directory.
