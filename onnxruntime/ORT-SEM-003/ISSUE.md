## Affected versions

- Latest stable release tested: `v1.30.0` (`f2c39fe2f838cf35ce7da92824f5a5e3ee6e88a7`)
- Default-branch build tested: `a11b4e593164c82f7ce95efce3693744103f8176`

## Summary

ORT_ENABLE_ALL terminates the process with SIGSEGV while EmbedLayerNormFusion inspects the now-runtime Range start value.

The source model loads with optimizations disabled. Enabling graph optimization reproduces the failure on ONNX Runtime 1.30.0. Disabling only `EmbedLayerNormFusion` removes the failure. A compiled 1.31 development wheel based on `a11b4e593164c82f7ce95efce3693744103f8176` reproduced the same root. The relevant implementation remains present at official main `96f73115c95968a3f31f2a110b33c164d847dda4`.

## Steps to reproduce

1. Clone the public reproducer and enter this case:

   ```bash
   git clone https://github.com/Yuhx141/graph-optimizer-issue-reproducers.git
   cd graph-optimizer-issue-reproducers/onnxruntime/ORT-SEM-003
   ```

2. Install the versions in [`requirements.txt`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxruntime/requirements.txt).
3. Run `python reproduce.py`.

The standalone script validates the source behavior, reproduces the optimized behavior, and disables the implicated transformer as a positive control. The readable reproducer is at https://github.com/Yuhx141/graph-optimizer-issue-reproducers/tree/main/onnxruntime/ORT-SEM-003.

## Expected result

The optimizer must preserve every public value, remaining consumer, graph interface and numerical result of the valid source model, or decline the rewrite.

## Actual result

ORT_ENABLE_ALL terminates the process with SIGSEGV while EmbedLayerNormFusion inspects the now-runtime Range start value.

## Likely cause and source location

the int64 IsInitializerWithExpectedValue overload receives null from GetConstantInitializer but constructs Initializer from *tensor_proto without the null check present in the float overload.

Relevant source locations:
- `onnxruntime/core/optimizer/utils.cc: int64_t IsInitializerWithExpectedValue`
- `onnxruntime/core/optimizer/embed_layer_norm_fusion.cc: MatchPositionEmbeddingSubgraphsFromGather`

## Environment

- OS: Ubuntu 24.04.4 LTS, x86-64
- CPU: Intel Core Ultra 5 225
- Python: 3.11.16
- NumPy: 2.4.6
- ONNX: 1.22.0
- ONNX Runtime: 1.30.0 (`f2c39fe2f838cf35ce7da92824f5a5e3ee6e88a7`)
- Execution provider: CPUExecutionProvider
- Optimization level: ORT_ENABLE_ALL

## Duplicate search

PR #27976 validates empty initializer contents. The int64 overload still lacks the float overload's null TensorProto check.
- https://github.com/microsoft/onnxruntime/pull/27976

## Reproducer files

- [`reproduce.py`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxruntime/ORT-SEM-003/reproduce.py): standalone self-check
- [`source.txt`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxruntime/ORT-SEM-003/source.txt): readable ONNX graph
- [`minimal_model.onnx`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxruntime/ORT-SEM-003/minimal_model.onnx): exact source model
- [`SHA256SUMS`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxruntime/ORT-SEM-003/SHA256SUMS): file hashes
