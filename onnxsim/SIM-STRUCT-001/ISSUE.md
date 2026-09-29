## What breaks

A legal `Conv -> Mul` graph uses a `[1,1,1,1]` one-element scale. The scale is a scalar under ONNX broadcasting, but after fusion the Conv bias becomes rank 4. ONNX Runtime then rejects the optimized Conv because its bias must be rank 1.

## Minimal reproduction

This is a one-node Conv/Mul reduction rather than the original larger graph.

```bash
git clone https://github.com/Yuhx141/graph-optimizer-issue-reproducers.git
cd graph-optimizer-issue-reproducers/onnxsim/SIM-STRUCT-001
python reproduce.py
```

- `singleton-rank scalar` — skip control: `fuse_mul_into_conv`

The script checks the source, runs the default simplifier in a child process, and repeats the case with only the named pass skipped. Full files and the recorded run are in [SIM-STRUCT-001](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/tree/main/onnxsim/SIM-STRUCT-001).

## Result

The source executes. The pass-disabled control is exact. The default optimized graph fails at the Conv with `bias must be a 1D tensor of size output_channels (2)`.

Skipping only `fuse_mul_into_conv` for the corresponding witness removes the failure or mismatch.

## Expected Conv bias

Folding a scalar output scale into Conv must keep the bias as a one-dimensional `[C]` tensor, or the pass must decline.

## Code path I traced

The scalar branch keeps `bias_scale = scale`, including the scale's original rank; only the per-channel branch reshapes it to one dimension. See [`fuse_mul_into_conv.h` lines 106-168](https://github.com/onnxsim/onnxsim/blob/cc035f51c7ee7b8363f8990d19c33562d827bfc4/onnxsim/passes/fuse_mul_into_conv.h#L106-L168) and the bias multiplication at [lines 198-204](https://github.com/onnxsim/onnxsim/blob/cc035f51c7ee7b8363f8990d19c33562d827bfc4/onnxsim/passes/fuse_mul_into_conv.h#L198-L204).

## Versions and prior search

- onnxsim master: `cc035f51c7ee7b8363f8990d19c33562d827bfc4`
- onnxsim/optimizer submodule: `8c13b168c37e74b610d0cccba4e9e01618c612a7`
- Latest release was checked separately where the affected pass exists; the attached result is authoritative for the master revision above.

The full issue/PR history of `onnxsim/onnxsim`, `onnxsim/optimizer`, and `onnx/optimizer` was searched on 2026-09-29.

- No exact report was found in all onnxsim/onnxsim issues and PRs or in onnxsim/optimizer and onnx/optimizer history.
- Related implementation PRs: [onnxsim/onnxsim#576](https://github.com/onnxsim/onnxsim/pull/576) and [onnx/optimizer#326](https://github.com/onnx/optimizer/pull/326). Neither covers a one-element tensor whose rank is greater than zero when a Conv bias is present.

## Reproducer material

The [reproducer directory](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/tree/main/onnxsim/SIM-STRUCT-001) contains:

- [`reproduce.py`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxsim/SIM-STRUCT-001/reproduce.py)
- [`cases.json`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxsim/SIM-STRUCT-001/cases.json)
- readable ONNX text and the exact small `.onnx` model(s)
- [`observed-current.json`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxsim/SIM-STRUCT-001/observed-current.json)
- [`SHA256SUMS`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxsim/SIM-STRUCT-001/SHA256SUMS)

## Environment

- Ubuntu 24.04.4 LTS, x86_64
- CPython 3.11.16
- ONNX 1.22.0
- ONNX Runtime 1.30.0, CPUExecutionProvider, graph optimization disabled for differential execution
- NumPy 2.4.6
