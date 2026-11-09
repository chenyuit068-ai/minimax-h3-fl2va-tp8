"""Experimental two-kernel block softmax, expressed and compiled only in Nine.

Not installed into the qualified v4 inference runtime. The exponent polynomial
and normal-range cutoff match v4; the reduction tree is different and must be
qualified independently before integration.
"""
import ninetoothed as nt
import ninetoothed.language as ntl
from ninetoothed import Tensor

BLOCK = 4096


def local_arrangement(scores, maxima, sums, output):
    return (scores.tile((1, BLOCK)), maxima.tile((1, 1)),
            sums.tile((1, 1)), output.tile((1, BLOCK)))


def local_application(scores, maxima, sums, output):
    value = scores*(1.4426950408889634*128**-0.5)
    value = ntl.where(scores.offsets(-1) < scores.source.shape[-1], value, float('-inf'))
    maximum = ntl.max(value, 1)
    exponent = value-maximum[:, None]
    bounded = ntl.maximum(exponent, -126.0)
    integral = ntl.floor(bounded)
    fraction = bounded-integral
    polynomial = 7.054911620801121e-09
    polynomial = 1.0178086009239696e-07+fraction*polynomial
    polynomial = 1.3215486790144305e-06+fraction*polynomial
    polynomial = 1.5252733804059838e-05+fraction*polynomial
    polynomial = 0.00015403530393381606+fraction*polynomial
    polynomial = 0.0013333558146428441+fraction*polynomial
    polynomial = 0.009618129107628477+fraction*polynomial
    polynomial = 0.055504108664821576+fraction*polynomial
    polynomial = 0.2402265069591007+fraction*polynomial
    polynomial = 0.6931471805599453+fraction*polynomial
    polynomial = 1.0+fraction*polynomial
    bits = ntl.cast(integral+127.0, ntl.int32) << 23
    power = ntl.cast(bits, ntl.float32, bitcast=True)
    numerator = ntl.where(exponent >= -126.0, polynomial*power, 0.0)
    maxima = maximum[:, None]
    sums = ntl.sum(numerator, 1)[:, None]
    output = numerator


def global_arrangement(local_values, maxima, sums, output):
    values = local_values.tile((1, BLOCK))
    maxima = maxima.tile((1, -1)).expand((-1, values.shape[1]))
    sums = sums.tile((1, -1)).expand((-1, values.shape[1]))
    return values, maxima, sums, output.tile((1, BLOCK))


def global_application(local_values, maxima, sums, output):
    maximum = ntl.max(maxima, 1)
    exponent = maxima-maximum[:, None]
    bounded = ntl.maximum(exponent, -126.0)
    integral = ntl.floor(bounded)
    fraction = bounded-integral
    polynomial = 7.054911620801121e-09
    polynomial = 1.0178086009239696e-07+fraction*polynomial
    polynomial = 1.3215486790144305e-06+fraction*polynomial
    polynomial = 1.5252733804059838e-05+fraction*polynomial
    polynomial = 0.00015403530393381606+fraction*polynomial
    polynomial = 0.0013333558146428441+fraction*polynomial
    polynomial = 0.009618129107628477+fraction*polynomial
    polynomial = 0.055504108664821576+fraction*polynomial
    polynomial = 0.2402265069591007+fraction*polynomial
    polynomial = 0.6931471805599453+fraction*polynomial
    polynomial = 1.0+fraction*polynomial
    bits = ntl.cast(integral+127.0, ntl.int32) << 23
    power = ntl.cast(bits, ntl.float32, bitcast=True)
    factors = ntl.where(exponent >= -126.0, polynomial*power, 0.0)
    denominator = ntl.sum(sums*factors, 1)
    block = ntl.min(local_values.offsets(-1), 0)//4096
    factor = ntl.sum(ntl.where(maxima.offsets(-1) == block, factors, 0.0), 1)
    output = local_values*(factor/denominator)[:, None]


local_kernel = nt.make(local_arrangement, local_application,
    (Tensor(2, other=float('-inf'), shape_options=(None, {'constexpr': True, 'upper_bound': 65536})),
     Tensor(2, shape_options=(None, {'constexpr': True, 'upper_bound': 16})),
     Tensor(2, shape_options=(None, {'constexpr': True, 'upper_bound': 16})),
     Tensor(2, shape_options=(None, {'constexpr': True, 'upper_bound': 65536}))),
    num_warps=4, num_stages=1, max_num_configs=1)
global_kernel = nt.make(global_arrangement, global_application,
    (Tensor(2, shape_options=(None, {'constexpr': True, 'upper_bound': 65536})),
     Tensor(2, other=float('-inf'), shape_options=(None, {'constexpr': True, 'upper_bound': 16})),
     Tensor(2, other=0, shape_options=(None, {'constexpr': True, 'upper_bound': 16})),
     Tensor(2, shape_options=(None, {'constexpr': True, 'upper_bound': 65536}))),
    num_warps=4, num_stages=1, max_num_configs=1)
