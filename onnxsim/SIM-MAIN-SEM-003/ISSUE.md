## DOUBLE epsilon is narrowed to zero

The source decompositions are DOUBLE and use epsilon `1e-300`. Both fusions accept the graph but narrow epsilon to a float attribute, which becomes zero and changes the result.

## Reproduction and measurements

There are separate LayerNormalization and RMSNormalization witnesses in the same package.

```bash
git clone https://github.com/Yuhx141/graph-optimizer-issue-reproducers.git
cd graph-optimizer-issue-reproducers/onnxsim/SIM-MAIN-SEM-003
python reproduce.py
```

- `LayerNormalization double epsilon` — skip control: `fuse_layer_norm`
- `RMSNormalization double epsilon` — skip control: `fuse_rms_norm`

The script checks the source, runs the default simplifier in a child process, and repeats the case with only the named pass skipped. Full files and the recorded run are in [SIM-MAIN-SEM-003](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/tree/main/onnxsim/SIM-MAIN-SEM-003).

Both reduced witnesses differ from their source by about `0.495163`; disabling the corresponding pass is exact.

Skipping only `fuse_layer_norm`, `fuse_rms_norm` for the corresponding witness removes the failure or mismatch.

## Expected conversion rule

A fusion must decline when the source epsilon cannot round-trip through the target attribute representation without changing semantics.

## Relevant conversions

LayerNorm extracts epsilon as double but explicitly casts it to float at [`fuse_layer_norm.h` lines 269-282](https://github.com/onnxsim/onnxsim/blob/cc035f51c7ee7b8363f8990d19c33562d827bfc4/onnxsim/passes/fuse_layer_norm.h#L269-L282). RMSNorm stores the match epsilon as float and converts a DOUBLE constant in `FetchScalarAsFloat`; see [`fuse_rms_norm.h` lines 114-150](https://github.com/onnxsim/onnxsim/blob/cc035f51c7ee7b8363f8990d19c33562d827bfc4/onnxsim/passes/fuse_rms_norm.h#L114-L150).

## Versions and duplicate check

- onnxsim master: `cc035f51c7ee7b8363f8990d19c33562d827bfc4`
- onnxsim/optimizer submodule: `8c13b168c37e74b610d0cccba4e9e01618c612a7`
- Latest release was checked separately where the affected pass exists; the attached result is authoritative for the master revision above.

The full issue/PR history of `onnxsim/onnxsim`, `onnxsim/optimizer`, and `onnx/optimizer` was searched on 2026-09-29.

- No exact report was found for DOUBLE epsilon narrowing.
- The original fusion PRs [onnxsim/onnxsim#647](https://github.com/onnxsim/onnxsim/pull/647) and [#656](https://github.com/onnxsim/onnxsim/pull/656) do not cover a DOUBLE source with epsilon below the float range.

## Files used

The [reproducer directory](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/tree/main/onnxsim/SIM-MAIN-SEM-003) contains:

- [`reproduce.py`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxsim/SIM-MAIN-SEM-003/reproduce.py)
- [`cases.json`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxsim/SIM-MAIN-SEM-003/cases.json)
- readable ONNX text and the exact small `.onnx` model(s)
- [`observed-current.json`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxsim/SIM-MAIN-SEM-003/observed-current.json)
- [`SHA256SUMS`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxsim/SIM-MAIN-SEM-003/SHA256SUMS)

- Ubuntu 24.04.4 LTS, x86_64
- CPython 3.11.16
- ONNX 1.22.0
- ONNX Runtime 1.30.0, CPUExecutionProvider, graph optimization disabled for differential execution
- NumPy 2.4.6
