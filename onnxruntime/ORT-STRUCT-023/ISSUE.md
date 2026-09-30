## Affected versions

- Latest stable release tested: `v1.30.0` (`f2c39fe2f838cf35ce7da92824f5a5e3ee6e88a7`)
- Also reproduced with a local 1.31.0 development CPU wheel.
- The affected implementation is unchanged on official `main` `b2097437af1184d2c0231fca2f2e6468158494b0`.

## Summary

With `session.enable_quant_qdq_cleanup=1`, `QDQFinalCleanupTransformer` can produce an invalid graph in two output-preservation cases:

1. `Q` is both a graph output and the input of `DQ`. The transformer removes Q/DQ and leaves graph output `Q` without a producer.
2. The value feeding Q has another consumer and DQ is a graph output. The transformer renames the source node output to the DQ graph-output name, but the other consumer still refers to the old name.

Both source models pass the ONNX checker and run with optimization disabled. Disabling only `QDQFinalCleanupTransformer` restores a valid optimized session.

## Reproduce

```bash
git clone https://github.com/Yuhx141/graph-optimizer-issue-reproducers.git
cd graph-optimizer-issue-reproducers/onnxruntime/ORT-STRUCT-023
python reproduce.py
```

The first model fails with:

```text
Failed to find node output or a constant initializer producing output: Q.
```

The second fails with:

```text
Invalid model. Node input 'R' is not a graph input, initializer, or output of a previous node.
```

## Expected result

QDQ cleanup should skip these patterns or preserve every graph output and other consumer when rewiring them.

## Likely cause

`CleanUpNodeSequence` checks whether each second node is a graph output, but not whether the first Q node is a graph output. Its graph-output shortcut also replaces the source node's output definition without updating other consumers of that source value:

https://github.com/microsoft/onnxruntime/blob/b2097437af1184d2c0231fca2f2e6468158494b0/onnxruntime/core/optimizer/qdq_transformer/qdq_final_cleanup.cc#L21-L138

The review discussion on #28793 independently notes the shared-source mutation concern inherited from this function. I did not find an issue or PR that reports either current failure with `session.enable_quant_qdq_cleanup`.

## Environment

- OS: Ubuntu 24.04.4 LTS, x86-64
- Python: 3.11
- NumPy: 2.4.6
- ONNX: 1.22.0
- ONNX Runtime: 1.30.0 release wheel and a local 1.31.0 development CPU wheel
- Execution provider: CPUExecutionProvider

## Reproducer

- https://github.com/Yuhx141/graph-optimizer-issue-reproducers/tree/main/onnxruntime/ORT-STRUCT-023
