"""Read original safetensors through bounded file mappings; no model computation.

This is the host storage building block for streaming. Native layer scheduling
and full-model residency are not implemented by this module.
"""
import hashlib
import json
import math
import mmap
from pathlib import Path
import struct
import torch

DTYPES={'F16':(torch.float16,2),'BF16':(torch.bfloat16,2),'F32':(torch.float32,4),
        'F64':(torch.float64,8),'I64':(torch.int64,8),'I32':(torch.int32,4)}


class MappedWeightStore:
    def __init__(self,directory,*,max_tensor_bytes=2*1024**3,expected_sha256=None):
        self.directory=Path(directory).resolve()
        self.max_tensor_bytes=max_tensor_bytes
        if max_tensor_bytes<=0:raise ValueError('max_tensor_bytes must be positive')
        indices=sorted(self.directory.glob('*.safetensors.index.json'))
        if len(indices)>1:raise ValueError('Ambiguous checkpoint indices')
        expected=json.loads(indices[0].read_text())['weight_map'] if indices else {}
        names=sorted(set(expected.values())) if expected else [p.name for p in sorted(self.directory.glob('*.safetensors'))]
        if not names:raise FileNotFoundError('No checkpoint shards')
        self.entries={};self.shards={}
        for name in names:
            path=(self.directory/name).resolve()
            if not path.is_relative_to(self.directory) or not path.is_file():
                raise ValueError('Checkpoint shard is outside its directory or missing')
            size=path.stat().st_size
            with path.open('rb') as file:
                prefix=file.read(8)
                if len(prefix)!=8:raise ValueError('Truncated checkpoint')
                length=struct.unpack('<Q',prefix)[0]
                if not 2<=length<=64*1024**2:raise ValueError('Invalid header length')
                raw=file.read(length)
                if len(raw)!=length:raise ValueError('Truncated checkpoint header')
                header=json.loads(raw)
            if expected_sha256 is not None:
                digest=expected_sha256.get(name)
                if not digest:raise ValueError(f'Missing official hash for {name}')
                with path.open('rb') as file:
                    if hashlib.file_digest(file,'sha256').hexdigest()!=digest:
                        raise ValueError(f'Checkpoint hash mismatch: {name}')
            self.shards[name]=(path,size)
            spans=[]
            for key,item in header.items():
                if key=='__metadata__':continue
                if key in self.entries:raise ValueError('Duplicate weight name')
                if expected and expected.get(key)!=name:raise ValueError('Index/shard mismatch')
                dtype,width=DTYPES[item['dtype']]
                shape=tuple(item['shape'])
                if any(type(d) is not int or d<0 for d in shape):raise ValueError('Invalid tensor shape')
                start,end=item['data_offsets'];count=math.prod(shape)
                if not 0<=start<=end<=size-8-length or end-start!=count*width:
                    raise ValueError('Invalid tensor byte span')
                spans.append((start,end))
                self.entries[key]=(name,8+length+start,count,shape,dtype,end-start)
            ordered=sorted(spans)
            if any(a[1]>b[0] for a,b in zip(ordered,ordered[1:])):
                raise ValueError('Overlapping weight byte spans')
        if expected and set(self.entries)!=set(expected):raise ValueError('Incomplete index coverage')
        self.hashes_verified=expected_sha256 is not None

    def tensor(self,name):
        shard,offset,count,shape,dtype,size=self.entries[name]
        if size>self.max_tensor_bytes:raise MemoryError('Weight exceeds configured mapping bound')
        path,original_size=self.shards[shard]
        if path.stat().st_size!=original_size:raise ValueError('Checkpoint changed after inspection')
        if count==0:return torch.empty(shape,dtype=dtype)
        aligned=offset//mmap.ALLOCATIONGRANULARITY*mmap.ALLOCATIONGRANULARITY
        delta=offset-aligned
        with path.open('rb') as file:
            # Private copy-on-write mapping: file bytes cannot be modified by callers.
            mapping=mmap.mmap(file.fileno(),delta+size,access=mmap.ACCESS_COPY,offset=aligned)
        # torch.frombuffer retains the mapping owner until all tensor views die.
        return torch.frombuffer(mapping,dtype=dtype,count=count,offset=delta).reshape(shape)
