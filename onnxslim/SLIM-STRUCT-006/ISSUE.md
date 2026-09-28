## Affected versions

- Latest stable release tested: `v0.1.96` (`1448291a6c6f4163b6216073eed87d76fcd03659`)
- Default-branch commit tested: `e877aedb25311c27abd044bbd9fae22d9a5ee69a`

## Summary

A legal training-mode BatchNormalization with three public outputs is fused into a Conv that is serialized with three outputs.

## Steps to reproduce

1. Clone the public reproducer repository and enter this case:

   ```bash
   git clone https://github.com/Yuhx141/graph-optimizer-issue-reproducers.git
   cd graph-optimizer-issue-reproducers/onnxslim/SLIM-STRUCT-006
   ```

2. Install the release dependencies listed in [`requirements.txt`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxslim/requirements.txt), or run against the tested source build.
3. Run `python reproduce.py`.

The script validates the source model, runs the implicated pass, disables only that pass as a positive control, and runs a one-condition negative control. The complete readable reproducer is at https://github.com/Yuhx141/graph-optimizer-issue-reproducers/tree/main/onnxslim/SLIM-STRUCT-006.

## Actual result

The target assigns mean and variance outputs to Conv and fails ONNX validation and backend loading.

## Expected result

The inference-only fusion must reject training-mode or multi-output BatchNormalization nodes.

## Why the source graph is legal

`source.onnx` passes `onnx.checker.check_model(..., full_check=True)` and strict ONNX shape inference. Its fixed input is in `input.json`, and it executes with ONNX Runtime graph optimizations disabled. Exact hashes are in `SHA256SUMS`.

## Pass attribution

The minimum pass is `FusionConvBN`. Disabling only that pass restores a valid graph with output exactly equal to the unoptimized source for this witness.

## One-condition control

`control.onnx` changes the critical condition while keeping the same operator family. It remains valid and exact after optimization; its input is `control-input.json`.

## Likely cause and source location

The matcher does not guard training_mode or the BatchNormalization output count before copying all outputs to Conv. See [onnxslim/core/pattern/fusion/convbn.py#L27-L79](https://github.com/inisis/OnnxSlim/blob/e877aedb25311c27abd044bbd9fae22d9a5ee69a/onnxslim/core/pattern/fusion/convbn.py#L27-L79).

## Duplicate search

No exact issue or PR found in a directed search performed on 2026-09-28.

## Environment

Linux x86-64, CPython 3.11, ONNX 1.22.0, NumPy 2.4.6, ONNX Runtime 1.30.0, CPU execution provider. Full revision and build details are in [environment.json](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/environment.json).

## Reproducer files

- [`reproduce.py`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxslim/SLIM-STRUCT-006/reproduce.py): direct pass-specific check
- [`source.txt`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxslim/SLIM-STRUCT-006/source.txt): readable ONNX graph
- [`control.txt`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxslim/SLIM-STRUCT-006/control.txt): one-condition control
- Tiny `.onnx` models, JSON inputs, and SHA-256 hashes are in the same directory.
