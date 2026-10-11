# ONNX Runtime CPU optimizer contracts

Install `onnx==1.22.0 numpy==2.4.6 onnxruntime==1.29.0`, then run `python reproduce.py run`.
Select one issue with `--candidate NAME`; choices are listed by `--help`.

The six reports have eight representatives (Cast naming has local/scope cases;
AveragePool checks opsets 13 and 18). Sources pass full ONNX checking and execute
with optimization disabled. The script asserts the reproduced failures and the
neighboring controls. Small binary models and fixed NumPy inputs are included;
`.onnx.txt` files show readable graphs. No private harness is required.

Also reproduced with the pinned official CPU nightly
`1.32.0.dev20261009005` (build `6e0f154`), Python 3.12.3 on Linux x86_64.
Neither wheel is claimed to be today's main. Native fixes have not been tested.
