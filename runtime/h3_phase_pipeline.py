"""Experimental one-component-at-a-time orchestration of the reference pipeline.

This module changes residency only. It does not replace model calculations.
Full official FL2VA validation is still required before submission.
"""
from contextlib import contextmanager
import gc
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace
import torch


@contextmanager
def deferred_parameters():
    previous=os.environ.get('INFINICORE_DEFER_PARAMETER_STORAGE')
    os.environ['INFINICORE_DEFER_PARAMETER_STORAGE']='1'
    try:yield
    finally:
        if previous is None:os.environ.pop('INFINICORE_DEFER_PARAMETER_STORAGE',None)
        else:os.environ['INFINICORE_DEFER_PARAMETER_STORAGE']=previous


class PhaseResidency:
    def __init__(self,*,runtime=None):
        self.factories={};self.active_name=None;self.active_model=None
        self.runtime=runtime;self.transitions=[]

    def register(self,name,factory):
        if name in self.factories:raise ValueError('Duplicate phase')
        self.factories[name]=factory

    def release(self):
        from infinilm.lib import _infinilm as native
        import infinicore as core
        core.sync_device();torch.cuda.synchronize()
        # Reclaim stream-owned scratch while the native engine still owns its
        # streams. Its destructor may destroy them when active_model is reset.
        if self.runtime is not None:self.runtime.clear_cache()
        self.active_model=None;self.active_name=None
        # The condition-encoder facade has a Python cycle around its native engine.
        gc.collect()
        torch.cuda.synchronize()
        native._h3_trim_memory();torch.cuda.empty_cache()

    def get(self,name):
        if name==self.active_name:return self.active_model
        self.release()
        started=time.perf_counter()
        record=dict(phase=name,status='LOADING')
        self.transitions.append(record)
        print('[h3-phase] '+json.dumps(record),flush=True)
        try:model=self.factories[name]()
        except BaseException as error:
            record.update(status='FAILED',error=f'{type(error).__name__}: {error}',load_seconds=time.perf_counter()-started)
            raise
        self.active_model=model;self.active_name=name
        record.update(status='LOADED',load_seconds=time.perf_counter()-started,torch_allocated_bytes=torch.cuda.memory_allocated())
        engine=getattr(model,'engine',None)
        if engine is not None and hasattr(engine,'_h3_mapped_weight_audit'):
            record['weight_loading']=engine._h3_mapped_weight_audit
        print('[h3-phase] '+json.dumps(record),flush=True)
        return model


class LazyVAE(torch.nn.Module):
    def __init__(self,name,directory,phases,device,dtype):
        super().__init__();self.name=name;self.phases=phases
        self.config=SimpleNamespace(**json.loads((Path(directory)/'config.json').read_text()))
        self.register_buffer('_device_anchor',torch.empty(0,device=device,dtype=dtype),persistent=False)
        if name=='vae':
            self.spatial_compression_ratio=math.prod(self.config.spatial_downsample_factors)
            self.temporal_compression_ratio=math.prod(self.config.temporal_downsample_factors)
            self.tokens_chunk_size=math.ceil(self.config.clip_length/self.temporal_compression_ratio)

    @property
    def device(self):return self._device_anchor.device
    @property
    def dtype(self):return self._device_anchor.dtype
    def encode(self,*args,**kwargs):return self.phases.get(self.name).encode(*args,**kwargs)
    def decode(self,*args,**kwargs):
        model=self.phases.get(self.name)
        if self.name=='vae':
            from h3_vae_attention import VideoVAEAttention
            with VideoVAEAttention() as attention:
                result=model.decode(*args,**kwargs)
                print('[h3-vae-nine] '+json.dumps(dict(calls=attention.calls,
                    peak_workspace_bytes=attention.peak_workspace_bytes)),flush=True)
                return result
        return model.decode(*args,**kwargs)


class LazyConditionModel(torch.nn.Module):
    def __init__(self,phases):super().__init__();self.phases=phases
    def forward(self,**kwargs):return self.phases.get('text_encoder').model(**kwargs)


class LazyConditionEncoder(torch.nn.Module):
    def __init__(self,directory,phases,device,dtype):
        super().__init__()
        from transformers import AutoConfig
        self.config=AutoConfig.from_pretrained(str(directory),local_files_only=True)
        self.model=LazyConditionModel(phases)
        self.register_buffer('_device_anchor',torch.empty(0,device=device,dtype=dtype),persistent=False)
    @property
    def device(self):return self._device_anchor.device
    @property
    def dtype(self):return self._device_anchor.dtype


class LazyH3Runner:
    def __init__(self,directory,phases):
        from infinilm.infer_engine import read_hf_config
        self.engine=SimpleNamespace(hf_config=read_hf_config(str(directory)))
        self.phases=phases
    def __call__(self,*args,**kwargs):return self.phases.get('transformer')(*args,**kwargs)


def make_pipeline(entry,paths,devices,timings,*,runtime=None,metadata_only=False):
    from diffusers import ModularPipeline,AutoencoderKLMiniMaxH3,AutoencoderKLMiniMaxH3Audio
    from infinilm.lib import _infinilm as native
    from infinilm.multimodal.minimax_h3 import (MiniMaxH3ConditionEncoder,
        MiniMaxH3TransformerRunner,MiniMaxH3DiffusersAdapter)
    from h3_mapped_loader import load_mapped_state_dict
    if len(devices)!=1:raise ValueError('Experimental phase scheduling supports one GPU')
    if not native._h3_streaming_storage_available():raise RuntimeError('Build streaming support first')
    device=torch.device('cuda',devices[0]);dtype=entry.DTYPE
    phases=PhaseResidency(runtime=runtime)
    root=Path(paths.partition_dir)
    if root.resolve()!=Path(paths.modular_dir).resolve():
        raise ValueError('The experimental phase loader supports the pinned modular root only')
    if not metadata_only:
        manifest=json.loads((root/'asset-download-manifest.json').read_text())
        if (manifest.get('repo')!='MiniMaxAI/MiniMax-H3' or
            manifest.get('revision')!='42ed227ee7df40d41602854ae760620d6eb651fe' or
            not manifest.get('complete')):
            raise ValueError('Complete pinned original assets are required')
        hashes={r['file']:r['sha256'] for r in manifest['files'] if r['file'].endswith('.safetensors')}
        if not all(r.get('verified_sha256')==r['sha256'] for r in manifest['files'] if r['file'].endswith('.safetensors')):
            raise ValueError('The manifest must record original shard hash verification')
    else:hashes={}
    def native_factory(name,constructor):
        if metadata_only:raise RuntimeError('Metadata inspection must not run model calculations')
        with deferred_parameters():
            model=constructor()
            prefix=name+'/'
            expected={filename[len(prefix):]:value for filename,value in hashes.items() if filename.startswith(prefix)}
            load_mapped_state_dict(model.engine,root/name,expected_sha256=expected)
        return model
    phases.register('text_encoder',lambda:native_factory('text_encoder',lambda:MiniMaxH3ConditionEncoder(
        root/'text_encoder',dtype=dtype,tp_device_ids=devices,load_weights=False)))
    phases.register('transformer',lambda:native_factory('transformer',lambda:MiniMaxH3TransformerRunner(
        root/'transformer',tp_device_ids=devices,load_weights=False)))
    def vae_factory(name,cls):
        if metadata_only:raise RuntimeError('Metadata inspection must not load model weights')
        model=cls.from_pretrained(str(Path(paths.modular_dir)/name),torch_dtype=dtype,
                                  local_files_only=True,low_cpu_mem_usage=True)
        model.requires_grad_(False);model.eval();return model.to(device=device,dtype=dtype)
    phases.register('vae',lambda:vae_factory('vae',AutoencoderKLMiniMaxH3))
    phases.register('audio_vae',lambda:vae_factory('audio_vae',AutoencoderKLMiniMaxH3Audio))
    class DeviceAdapter(MiniMaxH3DiffusersAdapter):
        @property
        def device(self):return device
    text=LazyConditionEncoder(root/'text_encoder',phases,device,dtype)
    transformer=DeviceAdapter(LazyH3Runner(root/'transformer',phases),dtype=dtype)
    pipe=ModularPipeline.from_pretrained(str(paths.modular_dir),local_files_only=True)
    pipe.update_components(text_encoder=text,transformer=transformer,
        vae=LazyVAE('vae',Path(paths.modular_dir)/'vae',phases,device,dtype),
        audio_vae=LazyVAE('audio_vae',Path(paths.modular_dir)/'audio_vae',phases,device,dtype))
    pipe.load_components(workflow='t2va',dtype=dtype,
                         pretrained_model_name_or_path=str(paths.modular_dir),local_files_only=True)
    pipe._h3_phase_residency=phases
    return pipe,text,transformer
