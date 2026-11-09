"""Synthetic host-only integrity, lifetime and mapping-bound checks."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import torch
from safetensors.torch import save_file
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from h3_weight_store import MappedWeightStore


def run():
    checks=[]
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);path=root/'model.safetensors'
        source={'half':torch.arange(12).reshape(3,4).half(),'bf16':torch.arange(8).bfloat16(),'float':torch.randn(17)}
        save_file(source,path)
        digest=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
        store=MappedWeightStore(root,expected_sha256={path.name:digest})
        for name,value in source.items():torch.testing.assert_close(store.tensor(name),value,rtol=0,atol=0)
        checks.append('original FP16/BF16/FP32 bytes and shapes preserved')
        retained=store.tensor('float');del store
        torch.testing.assert_close(retained,source['float'],rtol=0,atol=0)
        checks.append('tensor retains mapping after store deletion')
        retained.fill_(99)
        assert hashlib.file_digest(path.open('rb'),'sha256').hexdigest()==digest
        checks.append('copy-on-write cannot modify original checkpoint')
        try:MappedWeightStore(root,expected_sha256={path.name:'0'*64})
        except ValueError:checks.append('wrong official hash rejected')
        else:raise AssertionError('Hash mismatch accepted')
        store=MappedWeightStore(root,max_tensor_bytes=1)
        try:store.tensor('half')
        except MemoryError:checks.append('mapping memory bound enforced')
        else:raise AssertionError('Mapping bound ignored')
        (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'half':'../outside.safetensors'}}))
        try:MappedWeightStore(root)
        except ValueError:checks.append('invalid indexed path rejected')
        else:raise AssertionError('Invalid path accepted')
        del retained  # Release the mapping before Windows removes the temporary file.
    return dict(status='PASS',checks=checks,host_storage_only=True,native_offload_implemented=False,full_fl2va_tested=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    result=run();Path(args.output).write_text(json.dumps(result,indent=2));print(json.dumps(result))
