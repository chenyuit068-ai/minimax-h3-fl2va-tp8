"""Explicit Torch producer completion before native external-pointer handoff.

This scope changes host scheduling only. The original from_torch function still
retains the tensor owner and constructs the native wrapper; no values are cached.
"""
from collections import Counter
import functools
import threading
import time

_owner_lock=threading.Lock()
_owner=None

class TorchNativeHandoff:
    def __init__(self):
        self.calls=Counter();self.wait_seconds=0.;self.statistics_lock=threading.Lock()
        self.installed=False
    def __enter__(self):
        global _owner
        import infinicore
        import torch
        with _owner_lock:
            if _owner is not None:
                raise RuntimeError('A Torch/native handoff scope is already installed')
            self.core=infinicore;self.previous=infinicore.from_torch
            @functools.wraps(self.previous)
            def ordered(tensor):
                if tensor.is_cuda:
                    start=time.perf_counter()
                    torch.cuda.current_stream(tensor.device).synchronize()
                    elapsed=time.perf_counter()-start
                    with self.statistics_lock:
                        self.calls[str(tensor.device.index)]+=1
                        self.wait_seconds+=elapsed
                return self.previous(tensor)
            self.ordered=ordered
            infinicore.from_torch=ordered
            _owner=self;self.installed=True
        return self
    def snapshot(self):
        with self.statistics_lock:
            return dict(mode='torch_producer_stream_completion_before_native_pointer_handoff',
                waits_by_device=dict(self.calls),host_wait_seconds=self.wait_seconds,
                model_numerical_operations_changed=False,original_tensor_owner_retained=True)
    def __exit__(self,*error):
        global _owner
        with _owner_lock:
            if not self.installed or _owner is not self:
                raise RuntimeError('Torch/native handoff scope ownership changed')
            changed=self.core.from_torch is not self.ordered
            if not changed:self.core.from_torch=self.previous
            _owner=None;self.installed=False
        if changed:raise RuntimeError('Native tensor conversion changed during the handoff scope')
        return False
