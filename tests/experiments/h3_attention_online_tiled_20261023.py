"""Isolated NineToothed tile factory; FP32 online application stays unchanged."""
import ninetoothed as nt
from ninetoothed import Tensor
from h3_attention_online_candidate_20261023 import application


def make(block_q, block_k, warps=4):
    def arrangement(query, key, value, output):
        def query_or_output(t):
            t=t.tile((1,1,block_q,-1)).tile((1,1,1,1))
            t.dtype=t.dtype.squeeze((0,2,3))
            t.dtype.dtype=t.dtype.dtype.squeeze((0,1))
            return t
        query_arranged=query_or_output(query)
        def key_or_value(t):
            t=t.tile((1,1,block_k,-1)).tile((1,1,-1,-1)).expand((-1,-1,query_arranged.shape[-2],-1))
            t.dtype=t.dtype.squeeze((0,1,3))
            t.dtype.dtype=t.dtype.dtype.squeeze((0,1))
            return t
        return query_arranged,key_or_value(key),key_or_value(value),query_or_output(output)
    return nt.make(arrangement,application,
        tuple(Tensor(4,shape_options=(None,None,None,{'constexpr':True,'upper_bound':128})) for _ in range(4)),
        num_warps=warps,num_stages=1,max_num_configs=1)
