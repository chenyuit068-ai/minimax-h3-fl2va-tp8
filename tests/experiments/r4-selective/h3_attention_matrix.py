"""Selective R4 BF16 Q/K attention experiment.

A configurable fraction of matrix-attention calls retains original BF16 Q/K
inputs. The remaining calls use the validated R2 fused route. All device math
continues through NineToothed. This candidate is experimental until a complete
video comparison is recorded.
"""
import os
import torch
from h3_attention_pv_fused_candidate_20261023 import ChunkedMatrixAttention as _Fused
from h3_attention_native_qk import NativeInputQKAttention as _OriginalBF16

_CALL_INDEX = 0


def _use_bf16(call_index):
    every = int(os.environ.get('H3_R4_BF16_EVERY', '4'))
    offset = int(os.environ.get('H3_R4_BF16_OFFSET', '0'))
    if every <= 1:
        return True
    return call_index % every == offset % every


class ChunkedMatrixAttention:
    def __new__(cls, *tensors):
        global _CALL_INDEX
        call_index = _CALL_INDEX
        _CALL_INDEX += 1
        constructor = _Fused
        if tensors[0].dtype == torch.bfloat16 and _use_bf16(call_index):
            constructor = _OriginalBF16
        return constructor(*tensors)

    @staticmethod
    def required_workspace_bytes(n, d, dtype=None):
        every = int(os.environ.get('H3_R4_BF16_EVERY', '4'))
        coefficient = 12 if dtype == torch.bfloat16 and every <= 1 else 16
        padded = max(32, 1 << (n - 1).bit_length()) if n <= 4096 else ((n + 4095) // 4096) * 4096
        chunk = min(4096, padded)
        return coefficient * padded * d + 4 * chunk * padded + 12 * chunk * ((n + 4095) // 4096)
