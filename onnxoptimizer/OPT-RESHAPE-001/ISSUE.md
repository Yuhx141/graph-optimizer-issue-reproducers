## Affected versions

- Latest stable release tested: `v0.4.2` (`ed2d224364233dc724ddd94e0081dfd1e5df212a`)
- Default-branch commit tested: `c8d77f3b8326297d7bfb851513079616f7219cdc`

## Summary

Bypassing the first Reshape changes the reference input for a zero in the second shape.

## Steps to reproduce

1. Clone the public reproducer repository and enter this case:

   ```bash
   git clone https://github.com/Yuhx141/graph-optimizer-issue-reproducers.git
   cd graph-optimizer-issue-reproducers/onnxoptimizer/OPT-RESHAPE-001
   ```

2. Install the release dependencies listed in [`requirements.txt`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxoptimizer/requirements.txt), or run against the tested source build.
3. Run `python reproduce.py`.

The script validates the source model, runs the implicated pass, disables only that pass as a positive control, and runs a one-condition negative control. The complete readable reproducer is at https://github.com/Yuhx141/graph-optimizer-issue-reproducers/tree/main/onnxoptimizer/OPT-RESHAPE-001.

## Actual result

The optimized output has shape [2,3], and some variants also have inconsistent metadata.

## Expected result

[2,3] -> [3,2] -> [0,-1] must remain [3,2].

## Why the source graph is legal

`source.onnx` passes `onnx.checker.check_model(..., full_check=True)` and strict ONNX shape inference. Its fixed input is in `input.json`, and it executes with ONNX Runtime graph optimizations disabled. Exact hashes are in `SHA256SUMS`.

## Pass attribution

The minimum pass is `eliminate_consecutive_idempotent_ops`. Disabling only that pass restores a valid graph with output exactly equal to the unoptimized source for this witness.

## One-condition control

`control.onnx` changes the critical condition while keeping the same operator family. It remains valid and exact after optimization; its input is `control-input.json`.

## Likely cause and source location

The pass reconnects the second Reshape to the original input without translating zero dimensions. See [passes/eliminate_consecutive_idempotent_ops.h#L25-L48](https://github.com/onnx/optimizer/blob/c8d77f3b8326297d7bfb851513079616f7219cdc/passes/eliminate_consecutive_idempotent_ops.h#L25-L48).

## Duplicate search

No exact issue or PR found in a directed search performed on 2026-09-28.

## Environment

Linux x86-64, CPython 3.11, ONNX 1.22.0, NumPy 2.4.6, ONNX Runtime 1.30.0, CPU execution provider. Full revision and build details are in [environment.json](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/environment.json).

## Reproducer files

- [`reproduce.py`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxoptimizer/OPT-RESHAPE-001/reproduce.py): direct pass-specific check
- [`source.txt`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxoptimizer/OPT-RESHAPE-001/source.txt): readable ONNX graph
- [`control.txt`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxoptimizer/OPT-RESHAPE-001/control.txt): one-condition control
- Tiny `.onnx` models, JSON inputs, and SHA-256 hashes are in the same directory.
