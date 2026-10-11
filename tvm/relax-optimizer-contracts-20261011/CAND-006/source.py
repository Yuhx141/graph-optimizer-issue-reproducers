from __future__ import annotations

from tvm.script import ir as I
from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function(private=True)
    def lifted(value: R.Tensor((2,), dtype="float32"), env: R.Tensor((2,), dtype="float32"), unused: R.Tensor((2,), dtype="float32")) -> R.Tensor((2,), dtype="float32"):
        gv: R.Tensor((2,), dtype="float32") = R.add(value, env, ty=R.Tensor((2,), dtype="float32"))
        return gv

    @R.function
    def main(x: R.Tensor((2,), dtype="float32"), y: R.Tensor((2,), dtype="float32"), z: R.Tensor((2,), dtype="float32")) -> R.Tensor((2,), dtype="float32"):
        closure: R.Any = R.make_closure(Module.lifted, (x, z))
        gv1 = R.invoke_pure_closure(closure, (y,), ty_args=[R.Tensor((2,), dtype="float32")])
        return gv1
