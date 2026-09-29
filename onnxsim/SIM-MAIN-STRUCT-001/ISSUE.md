## Description

A constant-trip Loop contains an If whose branch graph captures the current loop-carried value. After unrolling, the copied If attribute still refers to the old loop-body name `v_in`, so the result has an unresolved input.

## Small Loop/If reproducer

The Loop has a constant trip count and the nested If captures the loop-carried value.

```bash
git clone https://github.com/Yuhx141/graph-optimizer-issue-reproducers.git
cd graph-optimizer-issue-reproducers/onnxsim/SIM-MAIN-STRUCT-001
python reproduce.py
```

- `nested If captures loop-carried value` — skip control: `eliminate_loop_with_const_trip_count`

The script checks the source, runs the default simplifier in a child process, and repeats the case with only the named pass skipped. Full files and the recorded run are in [SIM-MAIN-STRUCT-001](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/tree/main/onnxsim/SIM-MAIN-STRUCT-001).

## Optimized graph

The source and pass-disabled control return `[5.0]`. The default target is invalid: `v_in` is not produced by any previous node.

Skipping only `eliminate_loop_with_const_trip_count` for the corresponding witness removes the failure or mismatch.

## Expected handling

Every nested subgraph capture must be rewritten to the corresponding value for the current unrolled iteration, or the pass must decline when nested captures are present.

I may be missing a restriction on nested captures, but the source passes ONNX validation and executes before the rewrite. If this pattern is intentionally unsupported, declining the unroll seems safer than emitting an unresolved name.

## Where the stale name appears to come from

The unroller remaps direct node inputs through `value_dict`, but copies graph-valued attributes verbatim with `new_node->copyAttributes(*node)`. See [`eliminate_loop_with_const_trip_count.h` lines 222-275](https://github.com/onnxsim/onnxsim/blob/cc035f51c7ee7b8363f8990d19c33562d827bfc4/onnxsim/passes/eliminate_loop_with_const_trip_count.h#L222-L275).

## Related history

The full issue/PR history of `onnxsim/onnxsim`, `onnxsim/optimizer`, and `onnx/optimizer` was searched on 2026-09-29.

- [onnxsim/onnxsim#204](https://github.com/onnxsim/onnxsim/issues/204) is closely related because it discusses hidden outer-scope captures in Loop bodies, but it was closed after the reported graph change was attributed to Netron display behavior. This reproducer is an executable invalid graph caused by recursive capture remapping while a Loop is unrolled.
- No exact issue or fix for a nested If capturing a loop-carried value was found.

- onnxsim master: `cc035f51c7ee7b8363f8990d19c33562d827bfc4`
- onnxsim/optimizer submodule: `8c13b168c37e74b610d0cccba4e9e01618c612a7`
- Latest release was checked separately where the affected pass exists; the attached result is authoritative for the master revision above.

The [reproducer directory](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/tree/main/onnxsim/SIM-MAIN-STRUCT-001) contains:

- [`reproduce.py`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxsim/SIM-MAIN-STRUCT-001/reproduce.py)
- [`cases.json`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxsim/SIM-MAIN-STRUCT-001/cases.json)
- readable ONNX text and the exact small `.onnx` model(s)
- [`observed-current.json`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxsim/SIM-MAIN-STRUCT-001/observed-current.json)
- [`SHA256SUMS`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxsim/SIM-MAIN-STRUCT-001/SHA256SUMS)

- Ubuntu 24.04.4 LTS, x86_64
- CPython 3.11.16
- ONNX 1.22.0
- ONNX Runtime 1.30.0, CPUExecutionProvider, graph optimization disabled for differential execution
- NumPy 2.4.6
