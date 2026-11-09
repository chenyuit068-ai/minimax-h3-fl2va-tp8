"""CPU-only regression checks; this report cannot qualify GPU or model execution."""
import argparse
import ast
import collections
from contextlib import contextmanager, nullcontext, redirect_stderr
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
from unittest import mock

PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
sys.path.insert(0,str(PACKAGE/'runtime'))


def test_arguments():
    spec=importlib.util.spec_from_file_location('host_check_tp8_entry',PACKAGE/'run_eightcard.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    base=['run_eightcard.py','fl2va','--infinilm-root','unused','--model-path','unused',
          '--first-frame','unused.png','--text','A dog falls asleep','--output','unused.mp4']
    with mock.patch.object(sys,'argv',base):
        args,devices=module.parse_args()
        assert devices==list(range(8)) and args.steps==50 and args.num_frames==124
    rejected=0
    for extra in (['--gpus','0'],['--gpus','0,1,2,3,4,5,6,6'],['--gpus','-1,0,1,2,3,4,5,6'],
                  ['--steps','0'],['--height','928'],['--height','929','--width','768']):
        with mock.patch.object(sys,'argv',base+extra),redirect_stderr(io.StringIO()):
            try:module.parse_args()
            except SystemExit as error:assert error.code==2;rejected+=1
            else:raise AssertionError('Invalid generation arguments accepted: '+repr(extra))
    assert 'torch' not in sys.modules,'Host argument checks unexpectedly imported a device runtime'
    return dict(valid_whole_machine_defaults=True,rejected_invalid_cases=rejected)


def test_cache_and_warm_context():
    # Compile the actual production class without executing its GPU imports.
    # Only host bookkeeping and device-context calls are exercised by stubs.
    tree=ast.parse((PACKAGE/'runtime/h3_nine_runtime.py').read_text())
    definition=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='NativeFusions')
    current={'device':0,'stream':None}
    @contextmanager
    def device(index):
        previous=current['device'];current['device']=index
        try:yield
        finally:current['device']=previous
    @contextmanager
    def stream(value):
        previous=current['stream'];current['stream']=value
        try:yield
        finally:current['stream']=previous
    fake_torch=SimpleNamespace(cuda=SimpleNamespace(device=device,stream=stream,
        ExternalStream=lambda pointer,device:SimpleNamespace(pointer=pointer,device=device)),no_grad=nullcontext)
    namespace={'torch':fake_torch,'collections':collections}
    exec(compile(ast.Module(body=[definition],type_ignores=[]),'<production NativeFusions class>','exec'),namespace)
    cls=namespace['NativeFusions'];runtime=object.__new__(cls)
    runtime.device_ids=tuple(range(8));runtime.cache_policy='pinned'
    runtime.max_workspace_bytes=16;runtime.max_signatures=2
    runtime.cache=collections.OrderedDict();runtime.cache_devices={};runtime.cache_streams={}
    class Buffer:
        def numel(self):return 8
        def element_size(self):return 1
    for index in range(8):
        key=('scratch',index)
        runtime.cache[key]=SimpleNamespace(buffers=[Buffer()])
        runtime.cache_devices[key]=index
    def forbidden_evict(key):raise AssertionError('Pinned TP callback attempted eviction or synchronization')
    runtime._evict=forbidden_evict
    runtime._reserve_workspace(8,7) # Aggregate64 bytes; device7 uses only8 of16.
    try:runtime._reserve_workspace(9,7)
    except RuntimeError as error:assert 'Pinned Nine cache capacity' in str(error)
    else:raise AssertionError('Per-device capacity overflow accepted')
    signature=(2,7,1,1,512,5376)
    key=('residual',signature,None)
    launches=[]
    class Prepared:
        def _launch_native_pointers(self,pointers,stream):
            assert current['device']==7
            assert current['stream'].pointer==stream
            assert current['stream'].device==7
            launches.append((pointers,stream))
    runtime.cache=collections.OrderedDict([(key,Prepared())])
    runtime.closed=False;runtime.lock=threading.Lock()
    runtime.calls=collections.Counter();runtime.calls_by_device=collections.defaultdict(collections.Counter)
    runtime._run('residual',None,None,None,None,123,signature,(11,22,33,44))
    assert launches==[((11,22,33,44),123)] and current=={'device':0,'stream':None}
    assert runtime.calls_by_device[7]['residual']==1
    return dict(per_device_budget_independent=True,pinned_overflow_raises_without_eviction=True,
                warm_launch_selects_signature_device=True,caller_device_and_stream_restored=True,
                gpu_execution_tested=False)


def test_evidence():
    import h3_eightcard_evidence as evidence
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);identity={'host_fixture_only':True};items={}
        for name in evidence.REQUIRED_TESTS:
            path=root/(name+'.json');path.write_text(json.dumps(dict(status='PASS',world_size=8)))
            items[name]=dict(path=str(path),sha256=evidence.sha(path))
        record=dict(status='EIGHT_CARD_COMPONENTS_PASS',runtime_identity=identity,test_reports=items)
        path=root/'fixture-qualification.json'
        with mock.patch.dict(os.environ,{'H3_EIGHT_CARD_QUALIFICATION':str(path)}),mock.patch.object(
                evidence,'runtime_identity',return_value=identity):
            path.write_text(json.dumps(record))
            gate=evidence.validate_eightcard_report(PACKAGE)
            assert gate['official_similarity_validated'] is False and gate['full_fl2va_tested'] is False
            rejections=0
            changed=json.loads(json.dumps(record));changed['runtime_identity']={'changed':True}
            missing=json.loads(json.dumps(record));missing['test_reports'].pop('tiny_tp8_qwen')
            for invalid in (changed,missing):
                path.write_text(json.dumps(invalid))
                try:evidence.validate_eightcard_report(PACKAGE)
                except RuntimeError:rejections+=1
                else:raise AssertionError('Invalid qualification accepted')
            path.write_text(json.dumps(record))
            Path(items['collectives']['path']).write_text('tampered')
            try:evidence.validate_eightcard_report(PACKAGE)
            except RuntimeError:rejections+=1
            else:raise AssertionError('Changed component result accepted')
    return dict(rejected_changed_identity_incomplete_and_tampered_evidence=rejections,
                identity_was_host_fixture_not_gpu=True)


def test_original_view():
    import h3_eightcard_evidence as evidence
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);package=root/'package';view=root/'model-view'
        (package/'assets').mkdir(parents=True);(view/'transformer').mkdir(parents=True)
        target=view/'transformer/model.safetensors';target.write_bytes(b'host-only-fixture')
        row=dict(file='transformer/model.safetensors',bytes=target.stat().st_size,sha256=evidence.sha(target))
        expected=dict(repo='host-fixture',revision='host-fixture',files=[row])
        (package/'assets/original-model-files.json').write_text(json.dumps(expected))
        record=dict(expected,complete=True,files=[dict(row,verified_sha256=row['sha256'])])
        path=view/'asset-download-manifest.json';path.write_text(json.dumps(record))
        result=evidence.validate_original_model_view(view,package)
        assert result['full_hash_scan_repeated_during_generation'] is False
        count=0
        incomplete=dict(record,complete=False)
        wrong=json.loads(json.dumps(record));wrong['files'][0]['verified_sha256']='incorrect'
        duplicate=dict(record,files=record['files']*2)
        for invalid in (incomplete,wrong,duplicate):
            path.write_text(json.dumps(invalid))
            try:evidence.validate_original_model_view(view,package)
            except RuntimeError:count+=1
            else:raise AssertionError('Invalid original-weight verification accepted')
        path.write_text(json.dumps(record));target.write_bytes(b'changed-size')
        try:evidence.validate_original_model_view(view,package)
        except RuntimeError:count+=1
        else:raise AssertionError('Changed original-weight file size accepted')
    return dict(rejected_incomplete_hash_mismatch_duplicate_and_size_change=count,
                only_temporary_host_fixtures_used=True,official_weights_loaded=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-entry',type=Path,required=True)
    p.add_argument('--diffusers-source',type=Path,required=True)
    p.add_argument('--first-frame',type=Path,required=True)
    p.add_argument('--vae-config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Choose fresh CPU evidence')
    result=dict(status='RUNNING_HOST_ONLY',gpu_probe_executed=False,
                collective_operations_tested=False,native_multi_card_forward_tested=False,
                full_fl2va_tested=False,official_similarity_validated=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    def save():a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    save()
    try:
        result['arguments']=test_arguments()
        result['cache_policy']=test_cache_and_warm_context()
        result['evidence_gate']=test_evidence()
        result['original_asset_gate']=test_original_view()
        from h3_source_audit import capture
        result['source_identity']=capture(PACKAGE,a.reference_entry)
        # Execute the pinned reference's unmodified pure geometry helpers.
        from PIL import Image
        source=a.diffusers_source.read_text(encoding='utf-8');tree=ast.parse(source)
        selected=[]
        for node in tree.body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in (
                    'MINIMAX_H3_MIN_ASPECT_RATIO','MINIMAX_H3_MAX_ASPECT_RATIO') for t in node.targets):
                selected.append(node)
            if isinstance(node,ast.FunctionDef) and node.name in ('resolve_canvas_size','align_num_frames'):
                selected.append(node)
        namespace={};exec(compile(ast.Module(body=selected,type_ignores=[]),str(a.diffusers_source),'exec'),namespace)
        with Image.open(a.first_frame) as image:width,height=image.size
        resolved=namespace['resolve_canvas_size'](width,height,32,768,768*1344)
        config=json.loads(a.vae_config.read_text())
        chunks=config['clip_length'];tokens=math.ceil(chunks/math.prod(config['temporal_downsample_factors']))
        frames=namespace['align_num_frames'](124,chunks,tokens)
        result['uniform_case_metadata']=dict(image_width=width,image_height=height,
            image_sha256=hashlib.sha256(a.first_frame.read_bytes()).hexdigest(),
            prompt='A dog falls asleep',default_height=resolved[0],default_width=resolved[1],
            requested_num_frames=124,aligned_num_frames=frames,requested_steps=50,
            fps=24,nominal_seconds=frames/24,
            geometry_source_sha256=hashlib.sha256(source.encode('utf-8')).hexdigest(),
            target_platform_defaults_must_still_be_verified=True)
        for directory in (PACKAGE/'runtime',PACKAGE/'tests',PACKAGE/'scripts'):
            for path in directory.glob('*.py'):compile(path.read_bytes(),str(path),'exec')
        result['status']='PASS_HOST_ONLY_NOT_EIGHT_CARD_VALIDATION';save()
        print(json.dumps({k:v for k,v in result.items() if k!='source_identity'},ensure_ascii=False,indent=2))
    except BaseException as error:
        result.update(status='FAIL_HOST_CHECK',error=type(error).__name__+': '+str(error));save();raise


if __name__=='__main__':main()
