from __future__ import annotations

from tvm.script import ir as I
from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(a: R.Tensor((100, 3), dtype="float16"), b: R.Tensor((3, 4), dtype="float16"), c: R.Tensor((4, 2), dtype="float32")) -> R.Tensor((100, 2), dtype="float32"):
        inner: R.Tensor((100, 4), dtype="float32") = R.matmul(a, b, out_dtype="float32", ty=R.Tensor((100, 4), dtype="float32"))
        gv1: R.Tensor((100, 2), dtype="float32") = R.matmul(inner, c, out_dtype="float32", ty=R.Tensor((100, 2), dtype="float32"))
        return gv1
