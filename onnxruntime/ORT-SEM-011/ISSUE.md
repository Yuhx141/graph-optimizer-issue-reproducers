## Affected versions

- Latest stable release tested: `v1.30.0` (`f2c39fe2f838cf35ce7da92824f5a5e3ee6e88a7`)
- Default-branch build tested: `a11b4e593164c82f7ce95efce3693744103f8176`

## Summary

EmbedLayerNormFusion removes the producer of the newly public add3_out value; saving/building the optimized session fails because that graph output has no producer.

The source model loads with optimizations disabled. Enabling graph optimization reproduces the failure on ONNX Runtime 1.30.0. Disabling only `EmbedLayerNormFusion` removes the failure. A compiled 1.31 development wheel based on `a11b4e593164c82f7ce95efce3693744103f8176` reproduced the same root. The relevant implementation remains present at official main `96f73115c95968a3f31f2a110b33c164d847dda4`.

## Steps to reproduce

1. Clone the public reproducer and enter this case:

   ```bash
   git clone https://github.com/Yuhx141/graph-optimizer-issue-reproducers.git
   cd graph-optimizer-issue-reproducers/onnxruntime/ORT-SEM-011
   ```

2. Install the versions in [`requirements.txt`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxruntime/requirements.txt).
3. Run `python reproduce.py`.

The standalone script validates the source behavior, reproduces the optimized behavior, and disables the implicated transformer as a positive control. The readable reproducer is at https://github.com/Yuhx141/graph-optimizer-issue-reproducers/tree/main/onnxruntime/ORT-SEM-011.

## Expected result

The optimizer must preserve every public value, remaining consumer, graph interface and numerical result of the valid source model, or decline the rewrite.

## Actual result

EmbedLayerNormFusion removes the producer of the newly public add3_out value; saving/building the optimized session fails because that graph output has no producer.

## Likely cause and source location

FuseSubGraph checks output-edge counts but not whether each removed node output is a graph output, then unconditionally removes add_node and the other matched nodes.

Relevant source locations:
- `onnxruntime/core/optimizer/embed_layer_norm_fusion.cc: FuseSubGraph`

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

No exact issue or PR was found by pass-name, symptom and root-cause searches on 2026-09-29.
No exact issue or PR was found by pass-name, symptom and root-cause searches on 2026-09-29.

## Reproducer files

- [`reproduce.py`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxruntime/ORT-SEM-011/reproduce.py): standalone self-check
- [`source.txt`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxruntime/ORT-SEM-011/source.txt): readable ONNX graph
- [`minimal_model.onnx`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxruntime/ORT-SEM-011/minimal_model.onnx): exact source model
- [`SHA256SUMS`](https://github.com/Yuhx141/graph-optimizer-issue-reproducers/blob/main/onnxruntime/ORT-SEM-011/SHA256SUMS): file hashes
