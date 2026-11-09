"""Host-only ownership, ordering and cleanup checks; no device accuracy claim."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

path=Path(__file__).resolve().parents[1]/'runtime/h3_framework_streams.py'
spec=importlib.util.spec_from_file_location('handoff_under_test',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
events=[]
def previous(tensor):events.append('wrap');return tensor
core=SimpleNamespace(from_torch=previous)
stream=SimpleNamespace(synchronize=lambda:events.append('wait'))
torch=SimpleNamespace(cuda=SimpleNamespace(current_stream=lambda device:stream))
gpu=SimpleNamespace(is_cuda=True,device=SimpleNamespace(index=0))
cpu=SimpleNamespace(is_cuda=False)
with patch.dict(sys.modules,infinicore=core,torch=torch):
    with module.TorchNativeHandoff() as scope:
        assert core.from_torch(gpu) is gpu and events==['wait','wrap']
        events.clear();assert core.from_torch(cpu) is cpu and events==['wrap']
        assert scope.snapshot()['waits_by_device']=={'0':1}
        current=core.from_torch
        try:
            with module.TorchNativeHandoff():pass
        except RuntimeError:pass
        else:raise AssertionError('Nested installation was accepted')
        assert core.from_torch is current
    assert core.from_torch is previous
    try:
        with module.TorchNativeHandoff():raise ValueError('diagnostic exception')
    except ValueError:pass
    assert core.from_torch is previous and module._owner is None
print('PASS_HOST_HANDOFF_ORDERING_OWNERSHIP_RESTORATION_NOT_GPU_VALIDATION')
