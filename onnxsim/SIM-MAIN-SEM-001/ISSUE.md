## Question about the causal-mask contract

I am not sure whether the `-1e30` threshold is meant as an explicit approximation policy or as an exact semantic rewrite. The reduced model suggests the distinction is observable.

The pass accepts every off-diagonal mask value at or below `-1e30` as equivalent to the fused operator's hard causal mask. A finite additive penalty is not a hard mask: sufficiently large finite logits can overcome it.

## Reproducer

The logits are finite and chosen so that a finite additive penalty can be overcome.

```bash
git clone https://github.com/Yuhx141/graph-optimizer-issue-reproducers.git
cd graph-optimizer-issue-reproducers/onnxsim/SIM-MAIN-SEM-001
python reproduce.py
```

- `finite additive causal-looking mask` — skip control: `fuse_gqa`

The script checks the source, runs the default simplifier in a child process, and repeats the case with only the named pass skipped. Full files and the recorded run are in [SIM-MAIN-SEM-001](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/tree/main/onnxsim/SIM-MAIN-SEM-001).

## Observed difference

With a legal finite witness, source and target are finite but differ by a maximum absolute value of `5.0`; disabling only `fuse_gqa` is exact.

Skipping only `fuse_gqa` for the corresponding witness removes the failure or mismatch.

## Why I am asking

The fusion must preserve finite additive-mask semantics. It should require an exact representation of hard masking, preserve the mask, or decline.

`VerifyCausalMaskConstant` accepts finite values using the fixed threshold `-1e30f`, then the rewrite drops the source Add and relies on GroupQueryAttention's causal behavior. See [`fuse_gqa.h` lines 172-209](https://github.com/onnxsim/onnxsim/blob/cc035f51c7ee7b8363f8990d19c33562d827bfc4/onnxsim/passes/fuse_gqa.h#L172-L209).

Would requiring an actual hard-mask representation here be the intended fix, or is a bounded-input assumption expected for this fusion?

## Search notes and revision

The full issue/PR history of `onnxsim/onnxsim`, `onnxsim/optimizer`, and `onnx/optimizer` was searched on 2026-09-29.

- No issue or PR was found describing finite additive masks being reinterpreted as hard masks by `fuse_gqa`.

- onnxsim master: `cc035f51c7ee7b8363f8990d19c33562d827bfc4`
- onnxsim/optimizer submodule: `8c13b168c37e74b610d0cccba4e9e01618c612a7`
- Latest release was checked separately where the affected pass exists; the attached result is authoritative for the master revision above.

The [reproducer directory](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/tree/main/onnxsim/SIM-MAIN-SEM-001) contains:

- [`reproduce.py`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxsim/SIM-MAIN-SEM-001/reproduce.py)
- [`cases.json`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxsim/SIM-MAIN-SEM-001/cases.json)
- readable ONNX text and the exact small `.onnx` model(s)
- [`observed-current.json`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxsim/SIM-MAIN-SEM-001/observed-current.json)
- [`SHA256SUMS`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxsim/SIM-MAIN-SEM-001/SHA256SUMS)

- Ubuntu 24.04.4 LTS, x86_64
- CPython 3.11.16
- ONNX 1.22.0
- ONNX Runtime 1.30.0, CPUExecutionProvider, graph optimization disabled for differential execution
- NumPy 2.4.6
