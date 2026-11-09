"""H3 gated residual fusion: x + gate * branch, with materialized product rounding."""
import ninetoothed as nt
import ninetoothed.language as ntl
from ninetoothed import Tensor


@nt.jit(num_warps=4, num_stages=1, max_num_configs=1)
def residual_rows_8192(
    x: Tensor(3).tile((1, 1, 8192)),
    gate: Tensor(3).tile((1, 1, 8192)),
    branch: Tensor(3).tile((1, 1, 8192)),
    output: Tensor(3).tile((1, 1, 8192)),
):
    product = ntl.cast(ntl.cast(gate, ntl.float32) * ntl.cast(branch, ntl.float32), output.dtype)
    output = ntl.cast(x, ntl.float32) + ntl.cast(product, ntl.float32)
