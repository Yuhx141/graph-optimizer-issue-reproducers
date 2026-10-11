import tvm
from tvm import relax
from tvm.script import ir as I
from tvm.script import relax as R
from tvm import relax
from tvm.script import ir as I
from tvm.script import relax as R


@I.ir_module
class Before:
    @R.function(private=True)
    def lifted(
        value: R.Tensor([2], "float32"),
        env: R.Tensor([2], "float32"),
        unused: R.Tensor([2], "float32"),
    ):
        return R.add(value, env)

    @R.function
    def main(
        x: R.Tensor([2], "float32"),
        y: R.Tensor([2], "float32"),
        z: R.Tensor([2], "float32"),
    ):
        closure = R.make_closure(Before.lifted, (x, z))
        return R.invoke_pure_closure(
            closure, (y,), ty_args=R.Tensor([2], "float32")
        )
