"""Read checkpoint metadata without allocating model tensors or changing weights."""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def tensor_header(path):
    with path.open('rb') as f:
        prefix=f.read(8)
        if len(prefix)!=8: raise ValueError(f'Truncated safetensors: {path}')
        size=struct.unpack('<Q',prefix)[0]
        if not 2<=size<=64*1024*1024: raise ValueError(f'Invalid safetensors header size: {path}')
        raw=f.read(size)
        if len(raw)!=size: raise ValueError(f'Truncated safetensors header: {path}')
        data=json.loads(raw)
        end=path.stat().st_size-8-size
        for name,item in data.items():
            if name=='__metadata__':continue
            lo,hi=item['data_offsets']
            if not 0<=lo<=hi<=end:raise ValueError(f'Invalid tensor offset: {path}:{name}')
    return data


def inspect_checkpoint(path):
    result={'directory':str(path),'files':[],'weight_bytes':0}
    indices=sorted(path.glob('*.safetensors.index.json'))
    if len(indices)>1:raise ValueError(f'Ambiguous checkpoint indices in {path}')
    expected={}
    if indices:
        index=json.loads(indices[0].read_text())
        expected=index['weight_map']
        shards=[path/name for name in sorted(set(expected.values()))]
    else:shards=sorted(path.glob('*.safetensors'))
    if not shards:raise FileNotFoundError(f'No safetensors weights in {path}')
    observed=set()
    for shard in shards:
        if not shard.is_file():raise FileNotFoundError(f'Missing shard: {shard}')
        tensors=tensor_header(shard)
        for key,item in tensors.items():
            if key=='__metadata__':continue
            if key in observed:raise ValueError(f'Duplicate tensor {key}')
            if expected and expected.get(key)!=shard.name:raise ValueError(f'Index/shard mismatch: {key}')
            observed.add(key)
            result['weight_bytes']+=item['data_offsets'][1]-item['data_offsets'][0]
        result['files'].append({'file':shard.name,'bytes':shard.stat().st_size,'header_checked':True})
    if expected and observed!=set(expected):raise ValueError('Index tensor coverage mismatch')
    result['tensor_count']=len(observed)
    result['file_content_hash_checked']=False
    return result


def fl2va_model_directory(model_path):
    """Match the reference entry's FL2VA path aliases without importing models."""
    root=Path(model_path).expanduser().resolve()
    if root.name=='diffusers':return root
    if root.name in ('FL2VA','Ref2VA'):return root.parent/'diffusers'
    if (root/'diffusers').is_dir():return root/'diffusers'
    return root


def check(model_path, first_frame=None, device_check=True, gpus='0', memory_policy='resident'):
    if memory_policy not in ('resident','experimental_streaming'):raise ValueError('Unknown memory policy')
    root=fl2va_model_directory(model_path)
    result={'status':'BLOCKED','model_path':str(root),'errors':[],'components':{},
            'full_fl2va_tested':False,'offload_implemented':memory_policy=='experimental_streaming',
            'full_generation_offload_validated':False,'memory_policy':memory_policy}
    try:
        index=json.loads((root/'model_index.json').read_text())
        if index.get('_class_name')!='MiniMaxH3ModularPipeline':
            raise ValueError('Reference InfiniLM entry requires a MiniMaxH3ModularPipeline index')
        for name in ('text_encoder','transformer','vae','audio_vae'):
            # The modular index may name the visual component vae or video_vae.
            candidates=[root/name]
            if name=='vae':candidates.extend([root/'video_vae',root/'visual_vae'])
            component=next((p for p in candidates if p.is_dir()),None)
            if component is None:raise FileNotFoundError(f'Missing component: {name}')
            result['components'][name]=inspect_checkpoint(component)
        if first_frame is not None and not Path(first_frame).is_file():raise FileNotFoundError('First frame does not exist')
    except (OSError,ValueError,KeyError,TypeError) as exc:result['errors'].append(str(exc))
    if device_check:
        try:
            import torch
            if not torch.cuda.is_available():raise RuntimeError('CUDA/HIP device unavailable')
            selected=[int(i.strip()) for i in gpus.split(',')]
            if not selected or len(set(selected))!=len(selected) or min(selected)<0:
                raise ValueError('GPU list must contain distinct nonnegative indices')
            if max(selected)>=torch.cuda.device_count():raise ValueError('Requested GPU is unavailable')
            result['devices']=[dict(index=i,name=torch.cuda.get_device_name(i),
                                     total_bytes=torch.cuda.get_device_properties(i).total_memory)
                               for i in selected]
            result['hip']=torch.version.hip
            total=sum(d['total_bytes'] for d in result['devices'])
            transformer=result['components'].get('transformer',{}).get('weight_bytes',0)
            if memory_policy=='experimental_streaming':
                from infinilm.lib import _infinilm as native
                if len(selected)!=1 or not getattr(native,'_h3_streaming_storage_available',lambda:False)():
                    raise RuntimeError('Experimental streaming needs one GPU and the separate patched build')
                result['native_layer_storage_patch_loaded']=True
                result['warnings']=['Experimental phase scheduling has not passed full FL2VA validation']
            elif transformer>total:
                result['errors'].append('Resident transformer weights exceed aggregate GPU memory; the experimental streaming path requires an explicit opt-in and separate build')
            result['memory_estimate_is_lower_bound']=True
        except Exception as exc:result['errors'].append(str(exc))
    if not result['errors']:result['status']='PREFLIGHT_ONLY_PASS'
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model-path',required=True);p.add_argument('--first-frame')
    p.add_argument('--output',required=True);p.add_argument('--no-device-check',action='store_true');p.add_argument('--gpus',default='0');a=p.parse_args()
    r=check(a.model_path,a.first_frame,not a.no_device_check,a.gpus)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(r,ensure_ascii=False,indent=2));print(json.dumps(r,ensure_ascii=False))
    raise SystemExit(0 if r['status']=='PREFLIGHT_ONLY_PASS' else 2)
