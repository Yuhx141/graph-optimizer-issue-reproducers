from __future__ import annotations

from tvm.script import ir as I
from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((2, 2), dtype="float32"), weight: R.Tensor((2, 2), dtype="float32")) -> R.Tensor((2, 2), dtype="float32"):
        with R.dataflow():
            lv: R.Tensor((2, 2), dtype="float32") = R.permute_dims(weight)
            lv1: R.Tensor((2, 2), dtype="float32") = I.Call("relax.matmul", [x, lv], attrs=I.make_node("relax.attrs.MatmulAttrs", out_dtype=None), ty=R.Tensor((2, 2), dtype="float32"))
            gv: R.Tensor((2, 2), dtype="float32") = lv1
            R.output(gv)
        return gv
