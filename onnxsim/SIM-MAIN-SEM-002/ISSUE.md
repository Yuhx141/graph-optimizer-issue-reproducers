## Zero has two meanings here

Both matchers accept a source score multiplier of exactly zero and write `scale=0` into an ONNX Runtime contrib operator. In the target operators, zero is a sentinel for the default scale rather than the mathematical multiplier zero.

The source graphs use zero as an actual multiplier. The fused contrib operators use attribute value zero as “choose the default scale”.

## Two reduced cases

I included one Attention case and one GroupQueryAttention case because both write the same target attribute.

```bash
git clone https://github.com/Yuhx141/graph-optimizer-issue-reproducers.git
cd graph-optimizer-issue-reproducers/onnxsim/SIM-MAIN-SEM-002
python reproduce.py
```

- `Attention zero scale` — skip control: `fuse_attention`
- `GQA zero scale` — skip control: `fuse_gqa`

The script checks the source, runs the default simplifier in a child process, and repeats the case with only the named pass skipped. Full files and the recorded run are in [SIM-MAIN-SEM-002](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/tree/main/onnxsim/SIM-MAIN-SEM-002).

## Numerical result

The Attention witness differs by `6.5055866`; the GroupQueryAttention witness differs by `1.8620088`. In both cases disabling only the implicated pass is exact.

Skipping only `fuse_attention`, `fuse_gqa` for the corresponding witness removes the failure or mismatch.

## Expected behavior

A source multiplier of zero must remain zero semantically. If the target attribute cannot encode it, both passes must decline.

## Likely source of the mismatch

`MatchScaledQKMatMul` accepts any Mul scalar, including zero, at [`fuse_attention.h` lines 225-259](https://github.com/onnxsim/onnxsim/blob/cc035f51c7ee7b8363f8990d19c33562d827bfc4/onnxsim/passes/fuse_attention.h#L225-L259). Both rewrites write it directly to the contrib-op attribute: [`fuse_attention.h` line 754](https://github.com/onnxsim/onnxsim/blob/cc035f51c7ee7b8363f8990d19c33562d827bfc4/onnxsim/passes/fuse_attention.h#L754) and [`fuse_gqa.h` line 493](https://github.com/onnxsim/onnxsim/blob/cc035f51c7ee7b8363f8990d19c33562d827bfc4/onnxsim/passes/fuse_gqa.h#L493). ONNX Runtime's CPU implementation explicitly replaces `scale_ == 0.0f` with `1/sqrt(head_size)` ([official source](https://github.com/microsoft/onnxruntime/blob/96f73115c95968a3f31f2a110b33c164d847dda4/onnxruntime/contrib_ops/cpu/bert/attention_cpu_base.h#L118-L123)).

## Search and tested revision

The full issue/PR history of `onnxsim/onnxsim`, `onnxsim/optimizer`, and `onnx/optimizer` was searched on 2026-09-29.

- No exact issue was found for an explicit zero scale.
- The implementation PR [onnxsim/onnxsim#692](https://github.com/onnxsim/onnxsim/pull/692) covers nonzero Div/Mul scaling but not the target operator's zero-sentinel collision.

- onnxsim master: `cc035f51c7ee7b8363f8990d19c33562d827bfc4`
- onnxsim/optimizer submodule: `8c13b168c37e74b610d0cccba4e9e01618c612a7`
- Latest release was checked separately where the affected pass exists; the attached result is authoritative for the master revision above.

The [reproducer directory](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/tree/main/onnxsim/SIM-MAIN-SEM-002) contains:

- [`reproduce.py`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxsim/SIM-MAIN-SEM-002/reproduce.py)
- [`cases.json`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxsim/SIM-MAIN-SEM-002/cases.json)
- readable ONNX text and the exact small `.onnx` model(s)
- [`observed-current.json`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxsim/SIM-MAIN-SEM-002/observed-current.json)
- [`SHA256SUMS`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxsim/SIM-MAIN-SEM-002/SHA256SUMS)

- Ubuntu 24.04.4 LTS, x86_64
- CPython 3.11.16
- ONNX 1.22.0
- ONNX Runtime 1.30.0, CPUExecutionProvider, graph optimization disabled for differential execution
- NumPy 2.4.6
