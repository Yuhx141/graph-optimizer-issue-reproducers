# TVM Relax optimizer contracts

Use a source-built TVM with LLVM enabled at upstream revision
`27576ce2d1b2cf5212ad46d5710e594bdcbb5df0` and matching tvm-ffi submodule.
Tested TVM version: 0.28.dev0; Python 3.12.3; LLVM 18.1.3; Linux x86_64 CPU;
NumPy 2.5.4; apache-tvm-ffi-orcjit 0.1.2.
Configure the TVM Python package and built libraries as usual, then run:

```bash
python reproduce.py run
python reproduce.py run-one --candidate 002
```

The five passes are AdjustMatmulOrder (002), ExpandTupleArguments (005),
RemoveUnusedParameters (006), FoldBatchnormToConv2D (017), and
FuseTransposeMatmul (060). Each source is type checked and executed before
applying its named pass. Each failure has a neighboring control that executes
before and after optimization and satisfies rtol=1e-5, atol=1e-6.

Inputs are deterministic `arange(1, N+1)/10` tensors in the declared dtypes.
The BatchNorm source and its constants are constructed explicitly in
`batchnorm()` in the driver. The vector source is constructed in
`CAND-060/original-repro.py`. No private research harness is required.
The assertions describe these reproduced regressions; a future fix may make
the failure assertion stop reproducing. Native fixes have not been tested.
