"""Experimental bounded-score attention; all device math compiled by Nine.

Matmul arrangement adapted from ntops.kernels.mm (Apache-2.0).
No third-party precompiled numerical kernels are called.
"""
import torch
import ninetoothed as nt
import ninetoothed.language as ntl
from ninetoothed import Tensor
from h3_attention_staged import copy_rows
from h3_prepared_launch import PreparedNineToothed

def mm_arrangement(a,b,c):
    c=c.tile((32,128))
    a=a.tile((32,32)).tile((1,-1)).expand((-1,c.shape[1]))
    a.dtype=a.dtype.squeeze(0)
    b=b.tile((32,128)).tile((-1,1)).expand((c.shape[0],-1))
    b.dtype=b.dtype.squeeze(1)
    return a,b,c

def mm_application(a,b,c):
    accumulator=ntl.zeros(c.shape,dtype=ntl.float32)
    for k in range(a.shape[0]):
        accumulator=accumulator+ntl.dot(a[k],b[k],input_precision='ieee')
    c=accumulator

mm=nt.make(mm_arrangement,mm_application,tuple(Tensor(2) for _ in range(3)),num_warps=4,num_stages=1,max_num_configs=1)

def softmax_arrangement(scores,out):
    return scores.tile((1,-1)),out.tile((1,-1))

def softmax32(scores,out):
    value=scores*(1.4426950408889634*32**-0.5)
    value=ntl.where(scores.offsets(-1)<scores.source.shape[-1],value,float('-inf'))
    maximum=ntl.max(value,1)
    exponent=value-maximum[:,None]
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
    numerator=ntl.where(exponent>=-126.0,polynomial*power,0.0)
    out=numerator/ntl.sum(numerator,1)[:,None]

def softmax64(scores,out):
    value=scores*(1.4426950408889634*64**-0.5)
    value=ntl.where(scores.offsets(-1)<scores.source.shape[-1],value,float('-inf'))
    maximum=ntl.max(value,1)
    exponent=value-maximum[:,None]
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
    numerator=ntl.where(exponent>=-126.0,polynomial*power,0.0)
    out=numerator/ntl.sum(numerator,1)[:,None]

def softmax128(scores,out):
    value=scores*(1.4426950408889634*128**-0.5)
    value=ntl.where(scores.offsets(-1)<scores.source.shape[-1],value,float('-inf'))
    maximum=ntl.max(value,1)
    exponent=value-maximum[:,None]
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
    numerator=ntl.where(exponent>=-126.0,polynomial*power,0.0)
    out=numerator/ntl.sum(numerator,1)[:,None]

def softmax_kernel(dim,n):
    return nt.make(softmax_arrangement,{32:softmax32,64:softmax64,128:softmax128}[dim],
        (Tensor(2,other=float('-inf'),shape_options=(None,{'constexpr':True,'upper_bound':65536})),
         Tensor(2,shape_options=(None,{'constexpr':True,'upper_bound':65536}))),
        num_warps=4,num_stages=1,max_num_configs=1)

@nt.jit(num_warps=4,num_stages=1,max_num_configs=1)
def clear_buffer(output:Tensor(4).tile((1,1,1,256))):
    output=ntl.full(output.shape,0,dtype=output.dtype)


from h3_block_softmax_candidate_20261023 import local_kernel, global_kernel

class SplitSoftmax:
    def __init__(self, scores, output):
        rows, columns = scores.shape
        self.buffers = tuple(torch.empty((rows, (columns+4095)//4096), device=scores.device, dtype=torch.float32) for _ in range(2))
        self.pointers = tuple(t.data_ptr() for t in self.buffers)
        self.local = PreparedNineToothed(local_kernel, scores, *self.buffers, output)
        self.global_part = PreparedNineToothed(global_kernel, output, *self.buffers, output.view_as(output))
    def _launch_native_pointers(self, pointers, stream):
        self.local._launch_native_pointers((pointers[0], *self.pointers, pointers[1]), stream)
        self.global_part._launch_native_pointers((pointers[1], *self.pointers, pointers[1]), stream)

def qk_arrangement(a,b,c):
    c=c.tile((32,128))
    a=a.tile((32,16)).tile((1,-1)).expand((-1,c.shape[1]));a.dtype=a.dtype.squeeze(0)
    b=b.tile((16,128)).tile((-1,1)).expand((c.shape[0],-1));b.dtype=b.dtype.squeeze(1)
    return a,b,c

qk_mm=nt.make(qk_arrangement,mm_application,tuple(Tensor(2) for _ in range(3)),num_warps=4,num_stages=1,max_num_configs=1)

class ChunkedMatrixAttention:
    def __init__(self,*tensors):
        q,k,v,out=tensors
        self.specs=tuple((tuple(t.shape),tuple(t.stride()),t.dtype,t.device) for t in tensors)
        assert q.shape[0]==1 and all(t.shape==q.shape for t in tensors)
        self.heads=q.shape[1];self.n=q.shape[2];self.d=q.shape[3];self.padded_n=(max(32,1<<(self.n-1).bit_length()) if self.n<=4096 else ((self.n+4095)//4096)*4096);self.chunk=min(4096,self.padded_n)
        self.offsets=tuple(t.stride(1)*t.element_size() for t in tensors)
        self.buffers=tuple(torch.empty((1,1,self.padded_n,self.d),device=q.device,dtype=torch.float32) for _ in range(4))
        views=tuple(t[:,:1] for t in tensors)
        for buffer in self.buffers:clear_buffer(buffer)
        actual_buffers=tuple(buffer[:,:,:self.n] for buffer in self.buffers)
        self.inputs=tuple(PreparedNineToothed(copy_rows,a,b) for a,b in zip(views[:3],actual_buffers[:3]))
        self.output=PreparedNineToothed(copy_rows,actual_buffers[3],views[3])
        self.scores=torch.empty((self.chunk,self.padded_n),device=q.device,dtype=torch.float32)
        self.score_pointer=self.scores.data_ptr()
        self.pointers=tuple(t.data_ptr() for t in self.buffers)
        self.parts={}
        for size in (self.chunk,):
            a=self.buffers[0].view(self.padded_n,self.d)[:size]
            b=self.buffers[1].view(self.padded_n,self.d).t()
            c=self.scores[:size]
            qk=PreparedNineToothed(qk_mm,a,b,c)
            assert self.d==128, 'Candidate currently implements D128 only'
            softmax=SplitSoftmax(c[:,:self.n],c[:,:self.n].view_as(c[:,:self.n]))
            self.buffers=self.buffers+softmax.buffers
            self.parts[size]=(qk,softmax,PreparedNineToothed(mm,c,self.buffers[2].view(self.padded_n,self.d),self.buffers[3].view(self.padded_n,self.d)[:size]))

    def _launch_native_pointers(self,pointers,stream):
        for head in range(self.heads):
            current=tuple(p+head*s for p,s in zip(pointers,self.offsets))
            for i,kernel in enumerate(self.inputs):kernel._launch_native_pointers((current[i],self.pointers[i]),stream)
            for start in range(0,self.n,self.chunk):
                size=self.chunk
                qk,softmax,pv=self.parts[size]
                qk._launch_native_pointers((self.pointers[0]+start*self.d*4,self.pointers[1],self.score_pointer),stream)
                softmax._launch_native_pointers((self.score_pointer,self.score_pointer),stream)
                pv._launch_native_pointers((self.score_pointer,self.pointers[2],self.pointers[3]+start*self.d*4),stream)
            self.output._launch_native_pointers((self.pointers[3],current[3]),stream)

    def __call__(self,*tensors):
        assert tuple((tuple(t.shape),tuple(t.stride()),t.dtype,t.device) for t in tensors)==self.specs
        self._launch_native_pointers(tuple(t.data_ptr() for t in tensors),torch.cuda.current_stream().cuda_stream)
