"""Experimental native weight loading with retained host mapping ownership.

Only storage, naming, and the reference loader's dtype conversion happen here.
Forward calculations stay in the native model. Use a streaming-patched build.
"""
import os
from pathlib import Path
import torch
from h3_weight_store import MappedWeightStore

H3_PRESERVE_PREFIXES=('proj_in.','audio_proj_in.','time_embedder.','proj_out.',
                      'audio_proj_out.','video_patch_proj.','audio_patch_proj.',
                      'final_layer.video_out.','final_layer.audio_out.')


def load_mapped_state_dict(model,directory,*,dtype=None,expected_sha256=None):
    import infinicore as core
    from infinilm.lib import _infinilm as native
    from infinilm.modeling_utils import _WEIGHT_REMAPPER
    if not getattr(native,'_h3_streaming_storage_available',lambda:False)():
        raise RuntimeError('The loaded native library does not contain the streaming patch')
    if os.environ.get('INFINICORE_DEFER_PARAMETER_STORAGE')!='1':
        raise RuntimeError('Construct the native engine with deferred storage enabled first')
    model_type=model.hf_config.get('model_type','')
    if model_type not in ('minimax_h3','qwen3_vl'):
        raise ValueError('Only the two competition model types are supported')
    dist=getattr(model,'distributed_config',None)
    if dist is not None and (len(dist.tp_device_ids)!=1 or dist.pp_size!=1):
        raise ValueError('Mapped streaming is validated only for one device')
    if hasattr(model,'_h3_mapped_weight_owners'):
        raise RuntimeError('This engine already owns mapped weights')
    torch_dtype=core.utils.to_torch_dtype(dtype if dtype is not None else model.dtype)
    expected=set(model.state_dict_keyname())
    store=MappedWeightStore(directory,expected_sha256=expected_sha256)
    # from_torch borrows the raw pointer; these owners must survive the engine.
    # Attach BEFORE loading so an exception cannot leave dangling host pointers.
    owners=[];model._h3_mapped_weight_owners=owners
    loaded=set();skipped=[];mapped_bytes=0;converted_bytes=0
    remapper=_WEIGHT_REMAPPER.get(model_type)
    print(f'[h3-weights] Loading {model_type}: {len(expected)} native tensors',flush=True)
    for name in sorted(store.entries):
        if model_type=='qwen3_vl' and name not in expected:
            skipped.append(name);continue
        value=store.tensor(name)
        original_dtype=value.dtype
        preserve=model_type=='minimax_h3' and name.startswith(H3_PRESERVE_PREFIXES)
        if value.is_floating_point() and not preserve:value=value.to(dtype=torch_dtype)
        if value.dtype!=original_dtype:converted_bytes+=value.numel()*value.element_size()
        values=remapper({name:value},config=model.hf_config) if remapper else {name:value}
        for target,tensor in values.items():
            if target not in expected:
                skipped.append(target);continue
            if target in loaded:raise ValueError(f'Duplicate mapped target: {target}')
            tensor=tensor.contiguous()
            owners.append(tensor)
            model.load_state_dict({target:core.from_torch(tensor)},strict=False)
            core.sync_device()
            loaded.add(target);mapped_bytes+=tensor.numel()*tensor.element_size()
            if len(loaded)%100==0:
                print(f'[h3-weights] {model_type}: {len(loaded)}/{len(expected)} tensors',flush=True)
    missing=expected-loaded
    if missing:raise ValueError(f'Missing native weights: {sorted(missing)[:20]}')
    model.process_weights_after_loading()
    result=dict(status='LOADED_NOT_FORWARD_VALIDATED',model_type=model_type,
                tensor_count=len(loaded),retained_host_bytes=mapped_bytes,
                dtype_converted_bytes=converted_bytes,skipped_keys=skipped,
                original_shard_hashes_verified=store.hashes_verified,
                checkpoint_directory=str(Path(directory).resolve()),full_fl2va_tested=False)
    model._h3_mapped_weight_audit=result
    print(f'[h3-weights] {model_type}: all {len(loaded)} tensors loaded',flush=True)
    return result
