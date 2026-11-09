"""Opt-in native H3 fusion dispatch. No model output or input values are cached."""
import atexit
import collections
import json
import os
import threading
import time
import torch
from infinilm.lib import _infinilm
from h3_prepared_launch import PreparedNineToothed
from h3_adaln_ninetoothed import adaln_rows_8192, adaln_rows_8192_w8
from h3_residual_ninetoothed import residual_rows_8192
from h3_attention_ninetoothed import h3_attention,h3_attention_vector
from h3_attention_twopass import h3_attention_twopass
from h3_attention_staged import StagedAttention
from h3_platform_qualification import validate_hip_attention_platform


class NativeFusions:
    def __init__(self, max_signatures=32, *, pointwise=True, attention=False,
                 matrix_attention=False, max_workspace_bytes=1536*1024**2,
                 device_ids=None, cache_policy='lru'):
        if max_signatures < 1: raise ValueError('max_signatures must be positive')
        if max_workspace_bytes < 1:raise ValueError('max_workspace_bytes must be positive')
        if cache_policy not in ('lru','pinned'):raise ValueError('Unknown cache policy')
        self.device_ids=None if device_ids is None else tuple(device_ids)
        if self.device_ids is not None and (not self.device_ids or
                len(set(self.device_ids))!=len(self.device_ids) or
                any(type(i) is not int or i<0 for i in self.device_ids)):
            raise ValueError('device_ids must contain distinct nonnegative indices')
        if self.device_ids is not None and len(self.device_ids)>1 and cache_policy!='pinned':
            raise ValueError('Multi-device execution requires pinned scratch until forward completes')
        self.cache_policy=cache_policy
        if matrix_attention and (not attention or not torch.version.hip):
            raise ValueError('Experimental matrix attention requires HIP attention dispatch')
        self.platform_qualification = None
        if attention and torch.version.hip:
            arch=getattr(torch.cuda.get_device_properties(torch.cuda.current_device()),'gcnArchName','')
            if self.device_ids is not None:
                if os.environ.get('H3_PLATFORM_QUALIFICATION_MODE')=='component-tests':
                    self.platform_qualification=dict(status='UNQUALIFIED_EIGHT_DEVICE_COMPONENT_TESTS_ONLY',arch=arch)
                else:
                    from h3_eightcard_evidence import validate_eightcard_report
                    self.platform_qualification=validate_eightcard_report()
            else:
                self.platform_qualification = validate_hip_attention_platform(arch, torch.__version__)
        if torch.cuda.is_current_stream_capturing():raise RuntimeError('Graph capture is unsupported')
        self.max_signatures = max_signatures
        self.matrix_attention = matrix_attention
        self.max_workspace_bytes = max_workspace_bytes
        self.cache = collections.OrderedDict()
        self.cache_streams = {}
        self.cache_devices = {}
        self.peak_workspace_bytes = 0
        self.workspace_evictions = 0
        self.attention_routes = collections.Counter()
        self.started_at = time.monotonic()
        self.lock = threading.Lock()
        self.calls = collections.Counter()
        self.calls_by_device = collections.defaultdict(collections.Counter)
        self.closed = False
        for attr in ('_h3_to_dlpack', '_h3_set_nine_callback'):
            if not hasattr(_infinilm, attr): raise RuntimeError('Build the supplied InfiniLM integration patch first')
        _infinilm._h3_set_nine_callback(self._run,pointwise,attention)
        atexit.register(self.close)

    @staticmethod
    def _wrap(t):
        return torch.from_dlpack(_infinilm._h3_to_dlpack(t))

    @staticmethod
    def _workspace_bytes(prepared):
        buffers=list(getattr(prepared,'buffers',()))
        scores=getattr(prepared,'scores',None)
        if scores is not None:buffers.append(scores)
        return sum(t.numel()*t.element_size() for t in buffers)

    @property
    def workspace_bytes(self):
        return sum(self._workspace_bytes(p) for p in self.cache.values())

    def workspace_bytes_for_device(self,device):
        return sum(self._workspace_bytes(p) for key,p in self.cache.items()
                   if self.cache_devices[key]==device)

    def _evict(self,key):
        prepared=self.cache[key]
        if self._workspace_bytes(prepared):
            # A scratch allocation belongs to the native stream on which it
            # was created. Finish that stream before reclaiming its workspace.
            torch.cuda.ExternalStream(self.cache_streams[key],device=prepared.buffers[0].device).synchronize()
        del self.cache[key]
        self.cache_streams.pop(key,None)
        self.cache_devices.pop(key,None)
        self.workspace_evictions+=1

    def clear_cache(self):
        while self.cache:self._evict(next(iter(self.cache)))
        self.cache_streams.clear()
        self.cache_devices.clear()

    def _reserve_workspace(self,required,device):
        if required>self.max_workspace_bytes:
            raise RuntimeError('Attention workspace exceeds the configured byte budget')
        # In TP mode every rank has its own capacity. Never synchronize an old
        # stream during a TP callback: it may be waiting for another rank whose
        # callback cannot acquire the Python lock yet. Pinned mode fails before
        # allocation instead; explicit cleanup occurs after all ranks finish.
        def keys():
            return [k for k in self.cache if self.device_ids is None or self.cache_devices[k]==device]
        def used():
            return self.workspace_bytes if self.device_ids is None else self.workspace_bytes_for_device(device)
        candidates=keys()
        while len(candidates)>=self.max_signatures or used()+required>self.max_workspace_bytes:
            if self.cache_policy=='pinned':
                raise RuntimeError(f'Pinned Nine cache capacity exceeded on device {device}; '
                                   'increase its measured budget before the next run')
            self._evict(candidates[0]);candidates=keys()

    def _run(self, op, x, a, b, out, stream_pointer, signature, pointers):
        # The native engine releases the GIL and executes on a worker thread.
        # Run on its actual stream: no copies or device-wide barriers are needed.
        # Scratch buffers must never be shared between independent device streams.
        staged=op=='attention' and bool(torch.version.hip)
        key=(op,signature,stream_pointer if staged else None)
        device_index=int(signature[1])
        if self.device_ids is not None and device_index not in self.device_ids:
            raise RuntimeError('Native callback used an unselected device')
        # The warm path needs the same device/stream context as preparation.
        # Native worker threads may alternate devices between callbacks.
        with self.lock,torch.cuda.device(device_index),torch.no_grad(),torch.cuda.stream(
                torch.cuda.ExternalStream(stream_pointer,device=device_index)):
            if self.closed: raise RuntimeError('Nine runtime closed during forward')
            prepared=self.cache.get(key)
            if prepared is None:
                tensors=tuple(self._wrap(t) for t in (x,a,b,out))
                if op == 'attention':tensors=tuple(t.permute(0,2,1,3) for t in tensors)
                device=tensors[0].device
                if device.index!=device_index or any(t.device!=device for t in tensors):
                    raise RuntimeError('Native signature and DLPack device disagree')
                with torch.cuda.device(device),torch.no_grad(),torch.cuda.stream(torch.cuda.ExternalStream(stream_pointer,device=device)):
                    matrix=(staged and self.matrix_attention and tensors[0].shape[-1]==128 and
                            1024<=tensors[0].shape[-2]<=65536)
                    if staged:
                        n,d=tensors[0].shape[-2:]
                        padded=(max(32,1<<(n-1).bit_length()) if n<=4096 else ((n+4095)//4096)*4096)
                        required=(16*padded*d+4*min(4096,padded)*padded if matrix else 16*n*d)
                    else:required=0
                    if matrix:
                        from h3_attention_matrix import ChunkedMatrixAttention
                        required=ChunkedMatrixAttention.required_workspace_bytes(n,d)
                    self._reserve_workspace(required,device_index)
                    if op == 'adaln':
                        kernel = adaln_rows_8192_w8 if tensors[0].dtype == torch.float16 else adaln_rows_8192
                    elif op == 'residual': kernel = residual_rows_8192
                    elif op == 'attention':
                        kernel = h3_attention
                    else: raise ValueError('Unknown native operation')
                    # Preparation launches the current inputs; the call below
                    # repeats the same computation, never reuses a prior result.
                    if matrix:
                        from h3_attention_matrix import ChunkedMatrixAttention
                        prepared=ChunkedMatrixAttention(*tensors)
                    else:
                        prepared = (StagedAttention(h3_attention_twopass,*tensors) if staged else
                                    PreparedNineToothed(kernel, *tensors))
                    self.cache[key] = prepared
                    self.cache_streams[key]=stream_pointer
                    self.cache_devices[key]=device_index
                    used=self.workspace_bytes if self.device_ids is None else self.workspace_bytes_for_device(device_index)
                    assert used<=self.max_workspace_bytes
                    self.peak_workspace_bytes=max(self.peak_workspace_bytes,self.workspace_bytes)
            self.cache.move_to_end(key)
            prepared._launch_native_pointers(pointers,stream_pointer)
            self.calls[op] += 1
            self.calls_by_device[device_index][op] += 1
            if op=='attention':
                self.attention_routes['matrix' if hasattr(prepared,'scores') else 'vector' if staged else 'direct']+=1
                if self.calls[op]%16==0:
                    # This is host submission progress, not a synchronized
                    # timing result or proof of a completed denoising step.
                    print('[h3-nine-progress] '+json.dumps(dict(
                        submitted_attention_calls=self.calls[op],native_shape=list(signature[3:7]),
                        routes=dict(self.attention_routes),workspace_bytes=self.workspace_bytes,
                        elapsed_wall_seconds=time.monotonic()-self.started_at)),flush=True)

    def close(self):
        if not self.closed:
            _infinilm._h3_set_nine_callback(None)
            self.closed = True
            self.clear_cache()
            atexit.unregister(self.close)

    def __enter__(self): return self
    def __exit__(self, *args): self.close()
