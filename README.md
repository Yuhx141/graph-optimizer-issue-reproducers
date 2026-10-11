# Graph optimizer issue reproducers

Small public reproducers for upstream ONNX graph optimizer reports.

Each case contains a direct, pass-specific `reproduce.py`, a readable ONNX text graph, the tiny binary model used by the script, fixed JSON inputs, a one-condition control, and SHA-256 hashes. No model or input is embedded as base64.

Install the requirements for the relevant project, enter a case directory, and run:

```bash
python reproduce.py
```

The scripts are intentionally independent of the private research harness.

## CPU / Relax optimizer reports (2026-10-11)

- [ONNX Runtime CPU contracts](onnxruntime/cpu-optimizer-contracts-20261011): six reports, eight representatives.
- [TVM Relax optimizer contracts](tvm/relax-optimizer-contracts-20261011): five pass-specific reports.

These batches use a shared independent driver, fixed model/input fixtures or
explicit source builders, neighboring controls and SHA-256 manifests. See each
batch README for its environment and selection command.
