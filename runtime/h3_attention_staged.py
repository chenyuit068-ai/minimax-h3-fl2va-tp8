"""Bounded FP32 compatibility staging; all device arithmetic uses NineToothed."""
import torch
import ninetoothed as nt
import ninetoothed.language as ntl
from ninetoothed import Tensor
from h3_prepared_launch import PreparedNineToothed


@nt.jit(num_warps=4,num_stages=1,max_num_configs=1)
def copy_rows(source:Tensor(4).tile((1,1,1,256)),output:Tensor(4).tile((1,1,1,256))):
    output=ntl.cast(source,output.dtype)


class StagedAttention:
    def __init__(self,kernel,*tensors):
        q,k,v,out=tensors
        if q.ndim!=4 or q.shape[0]!=1 or not all(t.shape==q.shape and t.dtype==q.dtype and t.device==q.device for t in tensors):
            raise ValueError('Expected matching batch-one BHND tensors')
        self.heads=q.shape[1]
        self.head_offsets=tuple(t.stride(1)*t.element_size() for t in tensors)
        self.specs=tuple((tuple(t.shape),tuple(t.stride()),t.dtype,t.device) for t in tensors)
        shape=(1,1,q.shape[2],q.shape[3])
        self.buffers=tuple(torch.empty(shape,device=q.device,dtype=torch.float32) for _ in range(4))
        views=tuple(t[:,:1] for t in tensors)
        self.inputs=tuple(PreparedNineToothed(copy_rows,source,dest) for source,dest in zip(views[:3],self.buffers[:3]))
        self.attention=PreparedNineToothed(kernel,*self.buffers)
        self.output=PreparedNineToothed(copy_rows,self.buffers[3],views[3])
        self.buffer_pointers=tuple(t.data_ptr() for t in self.buffers)

    def _launch_native_pointers(self,pointers,stream):
        for head in range(self.heads):
            current=tuple(pointer+head*offset for pointer,offset in zip(pointers,self.head_offsets))
            for i,kernel in enumerate(self.inputs):
                kernel._launch_native_pointers((current[i],self.buffer_pointers[i]),stream)
            self.attention._launch_native_pointers(self.buffer_pointers,stream)
            self.output._launch_native_pointers((self.buffer_pointers[3],current[3]),stream)

    def __call__(self,*tensors):
        if tuple((tuple(t.shape),tuple(t.stride()),t.dtype,t.device) for t in tensors)!=self.specs:
            raise ValueError('Staged attention signature changed')
        self._launch_native_pointers(tuple(t.data_ptr() for t in tensors),torch.cuda.current_stream().cuda_stream)
