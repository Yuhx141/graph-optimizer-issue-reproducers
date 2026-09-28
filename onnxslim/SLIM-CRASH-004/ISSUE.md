## Affected versions

- Latest stable release tested: `v0.1.96` (`1448291a6c6f4163b6216073eed87d76fcd03659`)
- Default-branch commit tested: `e877aedb25311c27abd044bbd9fae22d9a5ee69a`

## Summary

Fusing BatchNormalization into a grouped ConvTranspose raises a NumPy broadcast exception.

## Steps to reproduce

1. Clone the public reproducer repository and enter this case:

   ```bash
   git clone https://github.com/Yuhx141/graph-optimizer-issue-reproducers.git
   cd graph-optimizer-issue-reproducers/onnxslim/SLIM-CRASH-004
   ```

2. Install the release dependencies listed in [`requirements.txt`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxslim/requirements.txt), or run against the tested source build.
3. Run `python reproduce.py`.

The script validates the source model, runs the implicated pass, disables only that pass as a positive control, and runs a one-condition negative control. The complete readable reproducer is at https://github.com/Yuhx141/graph-optimizer-issue-reproducers/tree/main/onnxslim/SLIM-CRASH-004.

## Actual result

Optimization raises a broadcast/reshape exception for group=2; group=1 is exact.

## Expected result

Grouped ConvTranspose output channels must be mapped correctly or the fusion must be skipped.

## Why the source graph is legal

`source.onnx` passes `onnx.checker.check_model(..., full_check=True)` and strict ONNX shape inference. Its fixed input is in `input.json`, and it executes with ONNX Runtime graph optimizations disabled. Exact hashes are in `SHA256SUMS`.

## Pass attribution

The minimum pass is `FusionConvTransposeBN`. Disabling only that pass restores a valid graph with output exactly equal to the unoptimized source for this witness.

## One-condition control

`control.onnx` changes the critical condition while keeping the same operator family. It remains valid and exact after optimization; its input is `control-input.json`.

## Likely cause and source location

The scale reshape treats weight axis 1 as the full output-channel axis and does not account for groups. See [onnxslim/core/pattern/fusion/convbn.py#L47-L52](https://github.com/inisis/OnnxSlim/blob/e877aedb25311c27abd044bbd9fae22d9a5ee69a/onnxslim/core/pattern/fusion/convbn.py#L47-L52).

## Duplicate search

No exact issue or PR found in a directed search performed on 2026-09-28.

## Environment

Linux x86-64, CPython 3.11, ONNX 1.22.0, NumPy 2.4.6, ONNX Runtime 1.30.0, CPU execution provider. Full revision and build details are in [environment.json](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/environment.json).

## Reproducer files

- [`reproduce.py`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxslim/SLIM-CRASH-004/reproduce.py): direct pass-specific check
- [`source.txt`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxslim/SLIM-CRASH-004/source.txt): readable ONNX graph
- [`control.txt`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxslim/SLIM-CRASH-004/control.txt): one-condition control
- Tiny `.onnx` models, JSON inputs, and SHA-256 hashes are in the same directory.

## Previous report

This replaces the withdrawn report [#352](https://github.com/inisis/OnnxSlim/issues/352) with a public, human-readable reproducer.
