from __future__ import annotations

from tvm.script import ir as I
from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((2,), dtype="float32"), y: R.Tensor((2,), dtype="float32")) -> R.Tensor((2,), dtype="float32"):
        closure: R.Any = R.make_closure(Module.pair_add, ())
        gv1 = R.invoke_pure_closure(closure, ((x, y),), ty_args=[R.Tensor((2,), dtype="float32")])
        return gv1

    @R.function(private=True)
    def pair_add(pair: R.Tuple(R.Tensor((2,), dtype="float32"), R.Tensor((2,), dtype="float32"))) -> R.Tensor((2,), dtype="float32"):
        gv: R.Tensor((2,), dtype="float32") = pair[0]
        gv1: R.Tensor((2,), dtype="float32") = pair[1]
        gv2: R.Tensor((2,), dtype="float32") = R.add(gv, gv1, ty=R.Tensor((2,), dtype="float32"))
        return gv2
