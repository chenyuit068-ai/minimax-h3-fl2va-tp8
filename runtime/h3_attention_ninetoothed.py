"""Noncausal H3 self attention expressed entirely in NineToothed.

Arrangement adapted from ntops' Apache-2.0 scaled_dot_product_attention.
Uses FP32 IEEE dot products and online softmax to support older DCUs.
This compatibility path has not been tuned for full video token counts.
"""
import ninetoothed as nt
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(query, key, value, output):
    def query_or_output(t):
        t=t.tile((1,1,16,-1)).tile((1,1,1,1))
        t.dtype=t.dtype.squeeze((0,2,3))
        t.dtype.dtype=t.dtype.dtype.squeeze((0,1))
        return t
    query_arranged=query_or_output(query)
    def key_or_value(t):
        t=t.tile((1,1,32,-1)).tile((1,1,-1,-1)).expand((-1,-1,query_arranged.shape[-2],-1))
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
            probabilities=ntl.exp2(scores-next_maximum[:,None])
            correction=ntl.exp2(maximum-next_maximum)
            acc=acc*correction[:,None]+ntl.dot(probabilities,ntl.cast(value[j],ntl.float32),input_precision='ieee')
            denominator=denominator*correction+ntl.sum(probabilities,1)
            maximum=next_maximum
        output[i]=acc/denominator[:,None]


h3_attention=nt.make(arrangement,application,tuple(Tensor(4,shape_options=(None,None,None,{'constexpr':True,'upper_bound':128})) for _ in range(4)),
                     num_warps=4,num_stages=1,max_num_configs=1)


def vector_arrangement(query,key,value,output):
    def query_or_output(t):
        t=t.tile((1,1,1,-1)).tile((1,1,1,1))
        t.dtype=t.dtype.squeeze((0,2,3))
        t.dtype.dtype=t.dtype.dtype.squeeze((0,1))
        return t
    arranged=query_or_output(query)
    def key_or_value(t):
        t=t.tile((1,1,1,-1)).tile((1,1,-1,-1)).expand((-1,-1,arranged.shape[-2],-1))
        t.dtype=t.dtype.squeeze((0,1,3))
        t.dtype.dtype=t.dtype.dtype.squeeze((0,1))
        return t
    return arranged,key_or_value(key),key_or_value(value),query_or_output(output)


def vector_application(query,key,value,output):
    for i in range(query.shape[0]):
        q=ntl.cast(query[i],ntl.float32)
        acc=ntl.zeros((q.shape[-1],),dtype=ntl.float32)
        denominator=ntl.full((),1,dtype=ntl.float32)
        maximum=ntl.full((),float('-inf'),dtype=ntl.float32)
        for j in range(key.shape[0]):
            scores=ntl.sum(q*ntl.cast(key[j],ntl.float32),1)*(1.4426950408889634*query.source.shape[-1]**-0.5)
            scores=ntl.where(key[j].offsets(-2)<key.source.shape[-2],scores,float('-inf'))
            next_maximum=ntl.maximum(maximum,ntl.max(scores,0))
            probabilities=ntl.exp2(scores-next_maximum)
            correction=ntl.exp2(maximum-next_maximum)
            acc=acc*correction+ntl.sum(probabilities[:,None]*ntl.cast(value[j],ntl.float32),0)
            denominator=denominator*correction+ntl.sum(probabilities,0)
            maximum=next_maximum
        output[i]=(acc/denominator)[None,:]


h3_attention_vector=nt.make(vector_arrangement,vector_application,
    tuple(Tensor(4,shape_options=(None,None,None,{'constexpr':True,'upper_bound':128})) for _ in range(4)),
    num_warps=4,num_stages=1,max_num_configs=1)
