"""NineToothed candidate for InfiniLM MiniMax-H3 apply_adaln.

Reference: unmodified Diffusers MiniMax-H3: x * (1 + scale) + shift.
Round scale+1 and the product separately, matching the original PyTorch operation order.
Used by the optional native bridge; full FL2VA validation is still required.
"""
import ninetoothed as nt
import ninetoothed.language as ntl
from ninetoothed import Tensor


@nt.jit(num_warps=4, num_stages=1, max_num_configs=1)
def adaln_1024(
    x: Tensor(3).flatten().tile((1024,)),
    scale: Tensor(3).flatten().tile((1024,)),
    shift: Tensor(3).flatten().tile((1024,)),
    output: Tensor(3).flatten().tile((1024,)),
):
    adjusted_scale = ntl.cast(1.0 + ntl.cast(scale, ntl.float32), output.dtype)
    product = ntl.cast(ntl.cast(x, ntl.float32) * ntl.cast(adjusted_scale, ntl.float32), output.dtype)
    output = ntl.cast(product, ntl.float32) + ntl.cast(shift, ntl.float32)


@nt.jit(num_warps=4, num_stages=1, max_num_configs=1)
def adaln_256(
    x: Tensor(3).flatten().tile((256,)),
    scale: Tensor(3).flatten().tile((256,)),
    shift: Tensor(3).flatten().tile((256,)),
    output: Tensor(3).flatten().tile((256,)),
):
    adjusted_scale = ntl.cast(1.0 + ntl.cast(scale, ntl.float32), output.dtype)
    product = ntl.cast(ntl.cast(x, ntl.float32) * ntl.cast(adjusted_scale, ntl.float32), output.dtype)
    output = ntl.cast(product, ntl.float32) + ntl.cast(shift, ntl.float32)


@nt.jit(num_warps=4, num_stages=1, max_num_configs=1)
def adaln_rows_256(
    x: Tensor(3).tile((1, 1, 256)),
    scale: Tensor(3).tile((1, 1, 256)),
    shift: Tensor(3).tile((1, 1, 256)),
    output: Tensor(3).tile((1, 1, 256)),
):
    adjusted_scale = ntl.cast(1.0 + ntl.cast(scale, ntl.float32), output.dtype)
    product = ntl.cast(ntl.cast(x, ntl.float32) * ntl.cast(adjusted_scale, ntl.float32), output.dtype)
    output = ntl.cast(product, ntl.float32) + ntl.cast(shift, ntl.float32)


@nt.jit(num_warps=4, num_stages=1, max_num_configs=1)
def adaln_rows_1024(
    x: Tensor(3).tile((1, 1, 1024)),
    scale: Tensor(3).tile((1, 1, 1024)),
    shift: Tensor(3).tile((1, 1, 1024)),
    output: Tensor(3).tile((1, 1, 1024)),
):
    adjusted_scale = ntl.cast(1.0 + ntl.cast(scale, ntl.float32), output.dtype)
    product = ntl.cast(ntl.cast(x, ntl.float32) * ntl.cast(adjusted_scale, ntl.float32), output.dtype)
    output = ntl.cast(product, ntl.float32) + ntl.cast(shift, ntl.float32)


@nt.jit(num_warps=4, num_stages=1, max_num_configs=1)
def adaln_rows_2048(
    x: Tensor(3).tile((1, 1, 2048)),
    scale: Tensor(3).tile((1, 1, 2048)),
    shift: Tensor(3).tile((1, 1, 2048)),
    output: Tensor(3).tile((1, 1, 2048)),
):
    adjusted_scale = ntl.cast(1.0 + ntl.cast(scale, ntl.float32), output.dtype)
    product = ntl.cast(ntl.cast(x, ntl.float32) * ntl.cast(adjusted_scale, ntl.float32), output.dtype)
    output = ntl.cast(product, ntl.float32) + ntl.cast(shift, ntl.float32)


@nt.jit(num_warps=4, num_stages=1, max_num_configs=1)
def adaln_rows_8192(
    x: Tensor(3).tile((1, 1, 8192)),
    scale: Tensor(3).tile((1, 1, 8192)),
    shift: Tensor(3).tile((1, 1, 8192)),
    output: Tensor(3).tile((1, 1, 8192)),
):
    adjusted_scale = ntl.cast(1.0 + ntl.cast(scale, ntl.float32), output.dtype)
    product = ntl.cast(ntl.cast(x, ntl.float32) * ntl.cast(adjusted_scale, ntl.float32), output.dtype)
    output = ntl.cast(product, ntl.float32) + ntl.cast(shift, ntl.float32)


@nt.jit(num_warps=4, num_stages=1, max_num_configs=1)
def adaln_rows_4096(
    x: Tensor(3).tile((1, 1, 4096)),
    scale: Tensor(3).tile((1, 1, 4096)),
    shift: Tensor(3).tile((1, 1, 4096)),
    output: Tensor(3).tile((1, 1, 4096)),
):
    adjusted_scale = ntl.cast(1.0 + ntl.cast(scale, ntl.float32), output.dtype)
    product = ntl.cast(ntl.cast(x, ntl.float32) * ntl.cast(adjusted_scale, ntl.float32), output.dtype)
    output = ntl.cast(product, ntl.float32) + ntl.cast(shift, ntl.float32)


@nt.jit(num_warps=8, num_stages=1, max_num_configs=1)
def adaln_rows_8192_w8(
    x: Tensor(3).tile((1, 1, 8192)),
    scale: Tensor(3).tile((1, 1, 8192)),
    shift: Tensor(3).tile((1, 1, 8192)),
    output: Tensor(3).tile((1, 1, 8192)),
):
    adjusted_scale = ntl.cast(1.0 + ntl.cast(scale, ntl.float32), output.dtype)
    product = ntl.cast(ntl.cast(x, ntl.float32) * ntl.cast(adjusted_scale, ntl.float32), output.dtype)
    output = ntl.cast(product, ntl.float32) + ntl.cast(shift, ntl.float32)
