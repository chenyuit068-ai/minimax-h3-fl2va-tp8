"""ISOLATED EXPERIMENT: software-exp2 online H3 self attention expressed entirely in NineToothed.

Arrangement adapted from ntops' Apache-2.0 scaled_dot_product_attention.
Uses FP32 IEEE dot products and online softmax to support older DCUs.
This compatibility path has not been tuned for full video token counts.
"""
import ninetoothed as nt
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(query, key, value, output):
    def query_or_output(t):
        t=t.tile((1,1,32,-1)).tile((1,1,1,1))
        t.dtype=t.dtype.squeeze((0,2,3))
        t.dtype.dtype=t.dtype.dtype.squeeze((0,1))
        return t
    query_arranged=query_or_output(query)
    def key_or_value(t):
        t=t.tile((1,1,64,-1)).tile((1,1,-1,-1)).expand((-1,-1,query_arranged.shape[-2],-1))
        t.dtype=t.dtype.squeeze((0,1,3))
        t.dtype.dtype=t.dtype.dtype.squeeze((0,1))
        return t
    return query_arranged,key_or_value(key),key_or_value(value),query_or_output(output)


def application(query,key,value,output):
    for i in range(query.shape[0]):
        q=ntl.cast(query[i],ntl.float32)
        acc=ntl.zeros((q.shape[-2],q.shape[-1]),dtype=ntl.float32)
        denominator=ntl.full((q.shape[-2],),1,dtype=ntl.float32)
        maximum=ntl.full((q.shape[-2],),float('-inf'),dtype=ntl.float32)
        for j in range(key.shape[0]):
            scores=ntl.dot(q,ntl.trans(ntl.cast(key[j],ntl.float32)),input_precision='ieee')
            scores=scores*(1.4426950408889634*query.source.shape[-1]**-0.5)
            scores=ntl.where(key[j].offsets(-2)<key.source.shape[-2],scores,float('-inf'))
            next_maximum=ntl.maximum(maximum,ntl.max(scores,1))
            exponent=scores-next_maximum[:,None]
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
            probabilities=ntl.where(exponent>=-126.0,polynomial*power,0.0)
            exponent=maximum-next_maximum
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
            correction=ntl.where(exponent>=-126.0,polynomial*power,0.0)
            acc=acc*correction[:,None]+ntl.dot(probabilities,ntl.cast(value[j],ntl.float32),input_precision='ieee')
            denominator=denominator*correction+ntl.sum(probabilities,1)
            maximum=next_maximum
        output[i]=acc/denominator[:,None]


h3_attention=nt.make(arrangement,application,tuple(Tensor(4,shape_options=(None,None,None,{'constexpr':True,'upper_bound':128})) for _ in range(4)),
                     num_warps=4,num_stages=1,max_num_configs=1)

