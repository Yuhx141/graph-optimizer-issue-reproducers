## Assertion while fusing RoPE

The reduced RoPE graph exposes both the final rotated tensor and the shared embedding used by the matched pattern as graph outputs. `fuse_rope` treats the embedding chain as dead and aborts in `eraseOutput` while destroying it.

The shared embedding is deliberately also a graph output, so it is still observable.

## Reproduce

The child process captures the native abort so the top-level script can report it cleanly.

```bash
git clone https://github.com/Yuhx141/graph-optimizer-issue-reproducers.git
cd graph-optimizer-issue-reproducers/onnxsim/SIM-MAIN-CRASH-001
python reproduce.py
```

- `public shared embedding` — skip control: `fuse_rope`

The script checks the source, runs the default simplifier in a child process, and repeats the case with only the named pass skipped. Full files and the recorded run are in [SIM-MAIN-CRASH-001](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/tree/main/onnxsim/SIM-MAIN-CRASH-001).

## What I see

The source and pass-disabled control execute and preserve both outputs. Default simplification aborts with `eraseOutput: Assertion outputs_[i]->uses().empty() failed`.

Skipping only `fuse_rope` for the corresponding witness removes the failure or mismatch.

## What I expected

A graph output is observable. The pass must preserve it or decline the fusion; simplification must not abort.

## Source reading

`MaybeAppendSharedChain` checks node-use counts but does not account for graph-output observability before appending nodes to the destruction list. See [`fuse_rope.h` lines 328-365](https://github.com/onnxsim/onnxsim/blob/cc035f51c7ee7b8363f8990d19c33562d827bfc4/onnxsim/passes/fuse_rope.h#L328-L365) and destruction at [lines 407-420](https://github.com/onnxsim/onnxsim/blob/cc035f51c7ee7b8363f8990d19c33562d827bfc4/onnxsim/passes/fuse_rope.h#L407-L420).

## Similar assertion reports

The full issue/PR history of `onnxsim/onnxsim`, `onnxsim/optimizer`, and `onnx/optimizer` was searched on 2026-09-29.

- [onnxsim/onnxsim#329](https://github.com/onnxsim/onnxsim/issues/329), [#341](https://github.com/onnxsim/onnxsim/issues/341), and [onnx/optimizer#39](https://github.com/onnx/optimizer/issues/39) contain the same generic `eraseOutput` assertion text. #341 was attributed to ONNX C++ IR name consistency and fixed through onnx/optimizer#277; the older reports involve other pass sets. None describes `fuse_rope` destroying a still-public intermediate.
- No exact RoPE/public-output report was found.

## Build used

- onnxsim master: `cc035f51c7ee7b8363f8990d19c33562d827bfc4`
- onnxsim/optimizer submodule: `8c13b168c37e74b610d0cccba4e9e01618c612a7`
- Latest release was checked separately where the affected pass exists; the attached result is authoritative for the master revision above.

The [reproducer directory](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/tree/main/onnxsim/SIM-MAIN-CRASH-001) contains:

- [`reproduce.py`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxsim/SIM-MAIN-CRASH-001/reproduce.py)
- [`cases.json`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxsim/SIM-MAIN-CRASH-001/cases.json)
- readable ONNX text and the exact small `.onnx` model(s)
- [`observed-current.json`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxsim/SIM-MAIN-CRASH-001/observed-current.json)
- [`SHA256SUMS`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxsim/SIM-MAIN-CRASH-001/SHA256SUMS)

- Ubuntu 24.04.4 LTS, x86_64
- CPython 3.11.16
- ONNX 1.22.0
- ONNX Runtime 1.30.0, CPUExecutionProvider, graph optimization disabled for differential execution
- NumPy 2.4.6
