import tvm
from tvm import relax
from tvm.script import ir as I
from tvm.script import relax as R
import tvm
from tvm import relax
from tvm.script import ir as I
from tvm.script import relax as R


@I.ir_module
class Before:
    @R.function(private=True)
    def pair_add(pair: R.Tuple(
        R.Tensor([2], "float32"), R.Tensor([2], "float32")
    )):
        return R.add(pair[0], pair[1])

    @R.function
    def main(x: R.Tensor([2], "float32"), y: R.Tensor([2], "float32")):
        closure = R.make_closure(Before.pair_add, ())
        return R.invoke_pure_closure(
            closure, ((x, y),), ty_args=R.Tensor([2], "float32")
        )
