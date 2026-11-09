"""Video-VAE SDPA adapter using the existing Nine FP32 matrix kernels only.

The reference processor still performs its original projections, RMSNorm and
rotary embedding. Only its unmasked, noncausal D64 attention dispatch changes.
Tensor permutations below are views; numeric attention runs through Nine.
"""
from collections import OrderedDict
import torch
from h3_attention_matrix_reference_v4 import ChunkedMatrixAttention


class VideoVAEAttention:
    def __init__(self):
        self.cache=OrderedDict();self.calls=0;self.peak_workspace_bytes=0

    def __enter__(self):
        from diffusers.models.autoencoders import autoencoder_kl_minimax_h3 as module
        self.module=module;self.previous=module.dispatch_attention_fn
        module.dispatch_attention_fn=self
        return self

    def __call__(self,query,key,value,*,attn_mask=None,is_causal=False,
                 backend=None,parallel_config=None,**kwargs):
        if attn_mask is not None or is_causal or parallel_config is not None or kwargs:
            raise ValueError('Video VAE Nine adapter supports only the original unmasked noncausal dispatch')
        if query.ndim!=4 or query.shape[-1]!=64 or query.shape[1]>4096:
            raise ValueError('Unqualified video VAE attention shape: '+str(tuple(query.shape)))
        if not query.is_cuda or query.dtype not in (torch.float16,torch.bfloat16):
            raise ValueError('Video VAE attention requires a qualified GPU half dtype')
        if any(t.shape!=query.shape or t.dtype!=query.dtype or t.device!=query.device for t in (key,value)):
            raise ValueError('Video VAE attention Q/K/V signatures differ')
        if backend is not None:
            raise ValueError('Explicit external attention backends are not supported')
        output=torch.empty(tuple(query.shape),device=query.device,dtype=query.dtype)
        for batch in range(query.shape[0]):
            tensors=tuple(t[batch:batch+1].permute(0,2,1,3) for t in (query,key,value,output))
            signature=tuple((tuple(t.shape),tuple(t.stride()),t.dtype,t.device) for t in tensors)
            if signature not in self.cache:
                if len(self.cache)>=4:
                    torch.cuda.current_stream().synchronize();self.cache.popitem(last=False)
                prepared=ChunkedMatrixAttention(*tensors)
                self.cache[signature]=prepared
                workspace=sum(t.numel()*t.element_size() for item in self.cache.values() for t in (*item.buffers,item.scores))
                if workspace>512*1024**2:raise RuntimeError('Video VAE Nine workspace exceeded 512 MiB')
                self.peak_workspace_bytes=max(self.peak_workspace_bytes,workspace)
            self.cache.move_to_end(signature)
            self.cache[signature](*tensors) # Supply current pointers on every call.
        self.calls+=1
        return output

    def __exit__(self,*args):
        self.module.dispatch_attention_fn=self.previous
        torch.cuda.current_stream().synchronize()
        self.cache.clear()
