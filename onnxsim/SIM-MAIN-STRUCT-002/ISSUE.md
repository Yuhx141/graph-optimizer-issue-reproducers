## Invalid Reshape produced by the simplifier

For an `Unsqueeze -> Flatten` chain with one known zero dimension and one symbolic dimension, the pass materializes the target shape as `[0,-1]` and sets `allowzero=1`. ONNX explicitly forbids combining a zero with `-1` when `allowzero=1`.

The source and pass-disabled graph execute with output shape `[0,12]`. Default simplification fails shape inference with `Invalid Target shape product of 0. Product cannot be 0 in combination with -1`.

## How to reproduce it

The source is valid and the input has a real zero-sized dimension.

```bash
git clone https://github.com/Yuhx141/graph-optimizer-issue-reproducers.git
cd graph-optimizer-issue-reproducers/onnxsim/SIM-MAIN-STRUCT-002
python reproduce.py
```

- `zero and symbolic output dimensions` — skip control: `fuse_reshape_family`

The script checks the source, runs the default simplifier in a child process, and repeats the case with only the named pass skipped. Full files and the recorded run are in [SIM-MAIN-STRUCT-002](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/tree/main/onnxsim/SIM-MAIN-STRUCT-002).

## Why this looks like a spec violation

The pass must decline when the inferred target contains both a literal zero and an unknown dimension, or encode a legal equivalent shape.

Unknown dimensions become `-1`, then any zero causes `allowzero=1`; the joint forbidden case is not checked. See [`fuse_reshape_family.h` lines 93-132](https://github.com/onnxsim/onnxsim/blob/cc035f51c7ee7b8363f8990d19c33562d827bfc4/onnxsim/passes/fuse_reshape_family.h#L93-L132). The [ONNX Reshape schema](https://onnx.ai/onnx/operators/onnx__Reshape.html) explicitly declares this combination invalid.

## Pass isolation

Skipping only `fuse_reshape_family` for the corresponding witness removes the failure or mismatch. The control leaves the `Unsqueeze -> Flatten` form intact and executes with shape `[0,12]`.

## Checked versions and related reports

- onnxsim master: `cc035f51c7ee7b8363f8990d19c33562d827bfc4`
- onnxsim/optimizer submodule: `8c13b168c37e74b610d0cccba4e9e01618c612a7`
- Latest release was checked separately where the affected pass exists; the attached result is authoritative for the master revision above.

The full issue/PR history of `onnxsim/onnxsim`, `onnxsim/optimizer`, and `onnx/optimizer` was searched on 2026-09-29.

- No exact report was found for `fuse_reshape_family` producing `[0,-1]` with `allowzero=1`.
- [onnx/optimizer#344](https://github.com/onnx/optimizer/issues/344) concerns consecutive Reshape elimination changing which input a zero copies from. It is a different pass and a different zero-dimension rule.
- The implementation PR [onnxsim/onnxsim#1030](https://github.com/onnxsim/onnxsim/pull/1030) discusses zeros and unknown dimensions independently, but has no test for their joint occurrence.

## Files

The [reproducer directory](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/tree/main/onnxsim/SIM-MAIN-STRUCT-002) contains:

- [`reproduce.py`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxsim/SIM-MAIN-STRUCT-002/reproduce.py)
- [`cases.json`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxsim/SIM-MAIN-STRUCT-002/cases.json)
- readable ONNX text and the exact small `.onnx` model(s)
- [`observed-current.json`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxsim/SIM-MAIN-STRUCT-002/observed-current.json)
- [`SHA256SUMS`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxsim/SIM-MAIN-STRUCT-002/SHA256SUMS)

- Ubuntu 24.04.4 LTS, x86_64
- CPython 3.11.16
- ONNX 1.22.0
- ONNX Runtime 1.30.0, CPUExecutionProvider, graph optimization disabled for differential execution
- NumPy 2.4.6
