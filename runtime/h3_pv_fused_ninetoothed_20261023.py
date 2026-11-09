"""Isolated Nine experiment: normalize probabilities inside the FP32 PV dot."""
import ninetoothed as nt
import ninetoothed.language as ntl
from ninetoothed import Tensor


def factor_arrangement(maxima,sums,output):
    return maxima.tile((1,-1)),sums.tile((1,-1)),output.tile((1,-1))


def factor_application(maxima,sums,output):
    maximum=ntl.max(maxima,1)
    exponent=maxima-maximum[:,None]
    bounded=ntl.maximum(exponent,-126.0)
    integral=ntl.floor(bounded)
    fraction=bounded-integral
    polynomial=7.054911620801121e-09
    polynomial=1.0178086009239696e-07+fraction*polynomial
    polynomial=1.3215486790144305e-06+fraction*polynomial
    polynomial=1.5252733804059838e-05+fraction*polynomial
    polynomial=0.00015403530393381606+fraction*polynomial
    polynomial=0.0013333558146428441+fraction*polynomial
    polynomial=0.009618129107628477+fraction*polynomial
    polynomial=0.055504108664821576+fraction*polynomial
    polynomial=0.2402265069591007+fraction*polynomial
    polynomial=0.6931471805599453+fraction*polynomial
    polynomial=1.0+fraction*polynomial
    bits=ntl.cast(integral+127.0,ntl.int32)<<23
    power=ntl.cast(bits,ntl.float32,bitcast=True)
    factors=ntl.where(exponent>=-126.0,polynomial*power,0.0)
    denominator=ntl.sum(sums*factors,1)
    output=factors/denominator[:,None]


factor_kernel=nt.make(factor_arrangement,factor_application,
    (Tensor(2,other=float('-inf'),shape_options=(None,{'constexpr':True,'upper_bound':16})),
     Tensor(2,other=0,shape_options=(None,{'constexpr':True,'upper_bound':16})),
     Tensor(2,shape_options=(None,{'constexpr':True,'upper_bound':16}))),
    num_warps=4,num_stages=1,max_num_configs=1)


def pv_arrangement(a,b,factors,c):
    c=c.tile((32,128))
    a=a.tile((32,32)).tile((1,-1)).expand((-1,c.shape[1]));a.dtype=a.dtype.squeeze(0)
    b=b.tile((32,128)).tile((-1,1)).expand((c.shape[0],-1));b.dtype=b.dtype.squeeze(1)
    factors=factors.tile((32,1)).tile((1,-1)).expand((-1,c.shape[1]))
    factors.dtype=factors.dtype.squeeze(0)
    return a,b,factors,c


def pv_application(a,b,factors,c):
    accumulator=ntl.zeros(c.shape,dtype=ntl.float32)
    for k in range(a.shape[0]):
        probabilities=a[k]*factors[k//128]
        accumulator=accumulator+ntl.dot(probabilities,b[k],input_precision='ieee')
    c=accumulator


pv_kernel=nt.make(pv_arrangement,pv_application,
    (Tensor(2),Tensor(2),Tensor(2,shape_options=(None,{'constexpr':True,'upper_bound':16})),Tensor(2)),
    num_warps=4,num_stages=1,max_num_configs=1)
