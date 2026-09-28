# Graph optimizer issue reproducers

Small public reproducers for upstream ONNX graph optimizer reports.

Each case contains a direct, pass-specific `reproduce.py`, a readable ONNX text graph, the tiny binary model used by the script, fixed JSON inputs, a one-condition control, and SHA-256 hashes. No model or input is embedded as base64.

Install the requirements for the relevant project, enter a case directory, and run:

```bash
python reproduce.py
```

The scripts are intentionally independent of the private research harness.
