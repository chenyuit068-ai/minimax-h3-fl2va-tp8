"""Use original BF16 Q/K with FP32 accumulation; preserve other dtype routes."""
import torch
from h3_attention_pv_fused_candidate_20261023 import ChunkedMatrixAttention as _Fused
from h3_attention_native_qk import NativeInputQKAttention as _OriginalBF16

class ChunkedMatrixAttention:
    def __new__(cls,*tensors):
        constructor=_OriginalBF16 if tensors[0].dtype==torch.bfloat16 else _Fused
        return constructor(*tensors)

    @staticmethod
    def required_workspace_bytes(n,d,dtype=None):
        padded=max(32,1<<(n-1).bit_length()) if n<=4096 else ((n+4095)//4096)*4096
        chunk=min(4096,padded)
        # The omitted-dtype API retains the conservative original bound.
        coefficient=12 if dtype==torch.bfloat16 else 16
        return coefficient*padded*d+4*chunk*padded+12*chunk*((n+4095)//4096)
