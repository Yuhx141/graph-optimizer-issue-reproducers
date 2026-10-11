import numpy as np
import tvm
from tvm import relax


def model(shape):
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorType(shape, "float32"))
    weight = relax.Var("weight", relax.TensorType(shape, "float32"))
    with bb.function("main", [x, weight]):
        with bb.dataflow():
            transposed = bb.emit(relax.op.permute_dims(weight))
            product = bb.emit(relax.op.matmul(x, transposed))
            output = bb.emit_output(product)
        bb.emit_func_output(output)
    return bb.finalize()


def execute(mod, data):
    executable = relax.build(mod, target="llvm", relax_pipeline="default")
    return relax.VirtualMachine(executable, tvm.cpu())["main"](
        tvm.runtime.tensor(data), tvm.runtime.tensor(data)
    ).numpy()


for shape in ((2,), (2, 2)):
    source = model(shape)
    data = np.arange(1, np.prod(shape) + 1, dtype="float32").reshape(shape)
    print("shape", shape, "source well formed", relax.analysis.check_well_formed(source, check_ty=True))
    print("source", execute(source, data).tolist())
    try:
        target = relax.transform.FuseTransposeMatmul()(source)
        print("target well formed", relax.analysis.check_well_formed(target, check_ty=True))
        print("target", execute(target, data).tolist())
    except Exception as error:
        print("pass error", type(error).__name__, str(error).splitlines()[0])
