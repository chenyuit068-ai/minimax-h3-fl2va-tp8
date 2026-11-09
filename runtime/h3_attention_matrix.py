"""Qualified PV-fused FP32 attention; exact workspace includes three block arrays."""
from h3_attention_pv_fused_candidate_20261023 import ChunkedMatrixAttention as _Fused

class ChunkedMatrixAttention(_Fused):
    @staticmethod
    def required_workspace_bytes(n, d):
        padded = max(32, 1 << (n-1).bit_length()) if n <= 4096 else ((n+4095)//4096)*4096
        chunk = min(4096, padded)
        return 16*padded*d + 4*chunk*padded + 12*chunk*((n+4095)//4096)
