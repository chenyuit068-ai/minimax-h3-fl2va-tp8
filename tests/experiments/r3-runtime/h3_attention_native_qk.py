"""Unqualified experiment: preserve original BF16/FP16 Q/K instead of upcasting.

Device math uses the existing Nine-generated dot, softmax, and PV kernels.
The formal inference entry does not import this file. Activate only after
independent numeric, actual-device, and complete-video comparisons.
"""
import torch
from h3_attention_pv_fused_candidate_20261023 import (
    ChunkedMatrixAttention as _Base, PreparedNineToothed, copy_rows,
    clear_buffer, mm, SplitSoftmax, pv_kernel)


class NativeInputQKAttention(_Base):
    @staticmethod
    def required_workspace_bytes(n,d):
        padded=max(32,1<<(n-1).bit_length()) if n<=4096 else ((n+4095)//4096)*4096
        chunk=min(4096,padded)
        return 12*padded*d+4*chunk*padded+12*chunk*((n+4095)//4096)

    def __init__(self,*tensors):
        q,k,v,out=tensors
        if q.dtype not in (torch.bfloat16,torch.float16) or k.dtype!=q.dtype:
            raise ValueError('Experiment requires the original 16-bit Q/K format; no inputs are quantized')
        self.specs=tuple((tuple(t.shape),tuple(t.stride()),t.dtype,t.device) for t in tensors)
        assert q.shape[0]==1 and all(t.shape==q.shape for t in tensors)
        self.heads=q.shape[1];self.n=q.shape[2];self.d=q.shape[3]
        assert self.d==128
        self.padded_n=max(32,1<<(self.n-1).bit_length()) if self.n<=4096 else ((self.n+4095)//4096)*4096
        self.chunk=min(4096,self.padded_n)
        self.offsets=tuple(t.stride(1)*t.element_size() for t in tensors)
        self.buffers=tuple(torch.empty((1,1,self.padded_n,self.d),device=q.device,dtype=dtype)
                           for dtype in (q.dtype,q.dtype,torch.float32,torch.float32))
        self.qk_element_bytes=self.buffers[0].element_size()
        views=tuple(t[:,:1] for t in tensors)
        for buffer in self.buffers:clear_buffer(buffer)
        actual=tuple(b[:,:,:self.n] for b in self.buffers)
        self.inputs=tuple(PreparedNineToothed(copy_rows,a,b) for a,b in zip(views[:3],actual[:3]))
        self.output=PreparedNineToothed(copy_rows,actual[3],views[3])
        self.scores=torch.empty((self.chunk,self.padded_n),device=q.device,dtype=torch.float32)
        self.score_pointer=self.scores.data_ptr();self.pointers=tuple(b.data_ptr() for b in self.buffers)
        c=self.scores
        qk=PreparedNineToothed(mm,self.buffers[0].view(self.padded_n,self.d)[:self.chunk],
                                self.buffers[1].view(self.padded_n,self.d).t(),c)
        softmax=SplitSoftmax(c[:,:self.n],c[:,:self.n].view_as(c[:,:self.n]))
        self.buffers=self.buffers+softmax.buffers
        self.parts={self.chunk:(qk,softmax,PreparedNineToothed(pv_kernel,c,
            self.buffers[2].view(self.padded_n,self.d),softmax.buffers[2],
            self.buffers[3].view(self.padded_n,self.d)[:self.chunk]))}

    def _launch_native_pointers(self,pointers,stream):
        for head in range(self.heads):
            current=tuple(p+head*s for p,s in zip(pointers,self.offsets))
            for i,kernel in enumerate(self.inputs):
                kernel._launch_native_pointers((current[i],self.pointers[i]),stream)
            for start in range(0,self.n,self.chunk):
                qk,softmax,pv=self.parts[self.chunk]
                qk._launch_native_pointers((self.pointers[0]+start*self.d*self.qk_element_bytes,
                                           self.pointers[1],self.score_pointer),stream)
                softmax._launch_native_pointers((self.score_pointer,self.score_pointer),stream)
                pv._launch_native_pointers((self.score_pointer,self.pointers[2],softmax.pointers[2],
                                           self.pointers[3]+start*self.d*4),stream)
            self.output._launch_native_pointers((self.pointers[3],current[3]),stream)
