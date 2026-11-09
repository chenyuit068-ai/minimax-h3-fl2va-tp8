"""Two-pass FP32 compatibility attention; all device math uses NineToothed."""
import ninetoothed as nt
import ninetoothed.language as ntl
from ninetoothed import Tensor

def arrangement(query,key,value,output):
    def q_or_out(t):
        t=t.tile((1,1,1,-1)).tile((1,1,1,1))
        t.dtype=t.dtype.squeeze((0,2,3))
        t.dtype.dtype=t.dtype.dtype.squeeze((0,1))
        return t
    arranged=q_or_out(query)
    def k_or_v(t):
        t=t.tile((1,1,32,-1)).tile((1,1,-1,-1)).expand((-1,-1,arranged.shape[-2],-1))
        t.dtype=t.dtype.squeeze((0,1,3))
        t.dtype.dtype=t.dtype.dtype.squeeze((0,1))
        return t
    return arranged,k_or_v(key),k_or_v(value),q_or_out(output)


def application(query,key,value,output):
    for i in range(query.shape[0]):
        q=ntl.cast(query[i],ntl.float32)
        maximum=ntl.full((),float('-inf'),dtype=ntl.float32)
        for j in range(key.shape[0]):
            scores=ntl.sum(q*ntl.cast(key[j],ntl.float32),1)*(1.4426950408889634*query.source.shape[-1]**-0.5)
            scores=ntl.where(key[j].offsets(-2)<key.source.shape[-2],scores,float('-inf'))
            maximum=ntl.maximum(maximum,ntl.max(scores,0))
        denominator=ntl.full((),0,dtype=ntl.float32)
        accumulator=ntl.zeros((q.shape[-1],),dtype=ntl.float32)
        for j in range(key.shape[0]):
            scores=ntl.sum(q*ntl.cast(key[j],ntl.float32),1)*(1.4426950408889634*query.source.shape[-1]**-0.5)
            scores=ntl.where(key[j].offsets(-2)<key.source.shape[-2],scores,float('-inf'))
            probabilities=ntl.exp2(scores-maximum)
            denominator=denominator+ntl.sum(probabilities,0)
            accumulator=accumulator+ntl.sum(probabilities[:,None]*ntl.cast(value[j],ntl.float32),0)
        output[i]=(accumulator/denominator)[None,:]



h3_attention_twopass=nt.make(arrangement,application,tuple(Tensor(4,shape_options=(None,None,None,{'constexpr':True,'upper_bound':128})) for _ in range(4)),num_warps=4,num_stages=1,max_num_configs=1)
