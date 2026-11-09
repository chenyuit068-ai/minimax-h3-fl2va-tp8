"""Run real allocated-DCU component tests and write evidence only on success."""
import argparse,datetime,hashlib,importlib.metadata as md,json,os,subprocess,sys,time
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--standalone',action='store_true')
p.add_argument('--attention-only',action='store_true')
p.add_argument('--reuse-subset',type=Path)
p.add_argument('--core-root',type=Path,required=True)
p.add_argument('--reference-entry',type=Path,required=True)
p.add_argument('--output-dir',type=Path,required=True)
args=p.parse_args()
package=Path(__file__).resolve().parents[1]
job=os.environ.get('SLURM_JOB_ID','direct-component-test')
output=args.output_dir.resolve()
output.mkdir(parents=True,exist_ok=False)
started=time.monotonic()
state=dict(status='RUNNING',job_id=job,full_fl2va_tested=False,
           started_at=datetime.datetime.now().astimezone().isoformat(),test_reports={})
statefile=output/'qualification-state.json'
def save():statefile.write_text(json.dumps(state,indent=2))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
save()
try:
    sys.path.insert(0,str(package/'runtime'))
    from h3_source_audit import capture
    capture(package,args.reference_entry)
    import torch,triton,ninetoothed
    assert torch.version.hip and torch.cuda.is_available(),'Allocated DCU unavailable to vendor Torch'
    assert torch.cuda.device_count()==1,'Expected exactly one Slurm-visible DCU'
    properties=torch.cuda.get_device_properties(0)
    arch=properties.gcnArchName
    assert arch.split(':')[0]=='gfx936',arch
    assert md.version('triton')=='3.3.0+das.opt1.dtk2604.torch290'
    state.update(arch=arch,device=torch.cuda.get_device_name(0),torch=torch.__version__,
                 torch_distribution=md.version('torch'),hip=torch.version.hip,
                 triton_distribution=md.version('triton'),ninetoothed=md.version('ninetoothed'),
                 total_device_memory=properties.total_memory,
                 visibility={n:os.environ.get(n) for n in ('HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','SLURM_JOB_GPUS')})
    save();print('ALLOCATED_DEVICE',json.dumps(state),flush=True)
    if not (args.standalone or args.attention_only):
        platform_manifest=package/'platform/op-cache-teardown-manifest.json'
        if platform_manifest.is_file():
            manifest=json.loads(platform_manifest.read_text())
            core_source=args.core_root.resolve()/'include/infinicore/ops/common/cache.hpp'
            assert sha(core_source)==manifest['after_sha256'],'Core cache teardown source is not patched'
            state['platform_native_sources']={str(core_source):sha(core_source),
                str(platform_manifest):sha(platform_manifest),
                str(package/'platform/InfiniCore-op-cache-teardown.patch'):sha(package/'platform/InfiniCore-op-cache-teardown.patch')}
        import infinicore
        from infinilm.lib import _infinilm
        assert _infinilm._h3_streaming_storage_available()
        state['native_framework_import']='PASS_ON_ALLOCATED_DCU'
        save();print('ALLOCATED_NODE_FRAMEWORK_IMPORT_PASS',flush=True)
    tests=package/'tests';experiments=tests/'experiments'
    commands={
      'native_matrix':[str(tests/'test_native_matrix_attention.py')],
      'tiny_h3':[str(tests/'test_streaming_tiny_h3.py'),'--matrix'],
      'tiny_qwen':[str(tests/'test_streaming_tiny_qwen.py'),'--matrix'],
      'pv_unit':[str(experiments/'test_pv_fused_unit_20261023.py'),'--runtime',str(package/'runtime')],
      'pv_attention':[str(experiments/'test_pv_fused_attention_20261023.py'),'--runtime',str(package/'runtime')],
      'block_softmax':[str(experiments/'test_block_softmax_target_20261023.py')],
    }
    env=dict(os.environ,H3_PLATFORM_QUALIFICATION_MODE='component-tests',PYTHONUNBUFFERED='1',PYTHONFAULTHANDLER='1')
    sys.path.insert(0,str(package/'runtime'))
    from h3_source_audit import inference_sources
    compiler=Path(triton.__file__).resolve().parent
    nine=Path(ninetoothed.__file__).resolve().parent
    files={'triton_hip_driver':compiler/'backends/hcu/driver.py',
           'triton_hip_abi':compiler/'backends/hcu/driver.c','nine_generation':nine/'generation.py'}
    compiler_records={n:dict(path=str(path),sha256=sha(path)) for n,path in files.items()}
    sources=inference_sources(package)
    state.update(inference_sources=sources,compiler_files=compiler_records,
                 attention_backend_manifest_sha256=sha(package/'attention-backend-manifest.json'))
    subset_names={'pv_unit','pv_attention','block_softmax'}
    if args.reuse_subset:
        previous=json.loads(args.reuse_subset.read_text())
        assert previous['status']=='COMPONENT_SUBSET_PASS'
        for name in ('arch','torch','triton_distribution','inference_sources','compiler_files','attention_backend_manifest_sha256'):
            assert previous[name]==state[name],('Previously tested platform changed',name)
        assert set(previous['test_reports'])==subset_names
        for name,item in previous['test_reports'].items():
            assert sha(item['path'])==item['sha256'] and json.loads(Path(item['path']).read_text())['status']=='PASS'
        state['test_reports'].update(previous['test_reports'])
        state['reused_component_evidence']=str(args.reuse_subset.resolve())
        if args.attention_only:state['test_reports'].pop('pv_attention')
    for name,command in commands.items():
        if (args.standalone or args.attention_only) and name not in subset_names:continue
        if name in state['test_reports']:continue
        report=output/(name+'.json');log=output/(name+'.log')
        state['stage']=name;save();print('COMPONENT_START',name,flush=True)
        t=time.monotonic()
        with log.open('w') as stream:
            result=subprocess.run([sys.executable,'-u',*command,'--output',str(report)],
                                  env=env,cwd=package,stdout=stream,stderr=subprocess.STDOUT,timeout=1800)
        assert result.returncode==0,(name,result.returncode,str(log))
        evidence=json.loads(report.read_text());assert evidence.get('status')=='PASS',(name,evidence.get('status'))
        state['test_reports'][name]=dict(path=str(report),sha256=sha(report),wall_seconds=time.monotonic()-t)
        save();print('COMPONENT_PASS',name,flush=True)
    if args.standalone or args.attention_only:
        assert set(state['test_reports'])==subset_names
        state.update(status='COMPONENT_SUBSET_PASS',stage='STANDALONE_NINE_TESTS_PASS',wall_seconds=time.monotonic()-started)
        (output/'component-subset.json').write_text(json.dumps(state,indent=2));save()
        print('STANDALONE_NINE_COMPONENT_SUBSET_PASS',str(output/'component-subset.json'),flush=True)
        raise SystemExit(0)
    native=Path(os.environ['INFINI_ROOT'])/'lib'
    names=('libinfiniccl.so','libinfiniop.so','libinfinirt.so','libinfinicore_cpp_api.so')
    libraries={str(native/n):sha(native/n) for n in names}
    compiler=Path(triton.__file__).resolve().parent
    nine=Path(ninetoothed.__file__).resolve().parent
    files={'triton_hip_driver':compiler/'backends/hcu/driver.py',
           'triton_hip_abi':compiler/'backends/hcu/driver.c','nine_generation':nine/'generation.py'}
    state.update(status='PASS',stage='ALL_COMPONENTS_PASS',inference_sources=inference_sources(package),
                 attention_backend_manifest_sha256=sha(package/'attention-backend-manifest.json'),
                 native_libraries=libraries,
                 compiler_files={n:dict(path=str(p),sha256=sha(p)) for n,p in files.items()},
                 wall_seconds=time.monotonic()-started,finished_at=datetime.datetime.now().astimezone().isoformat())
    qualification=output/'platform-qualification.json';qualification.write_text(json.dumps(state,indent=2))
    os.environ.pop('H3_PLATFORM_QUALIFICATION_MODE',None)
    os.environ['H3_PLATFORM_QUALIFICATION']=str(qualification)
    from h3_platform_qualification import validate_hip_attention_platform
    gate=validate_hip_attention_platform(arch,torch.__version__,package)
    assert gate['status']=='GFX936_COMPONENT_QUALIFICATION_PASS'
    save();print('PLATFORM_QUALIFICATION_PASS',str(qualification),json.dumps(gate),flush=True)
except Exception as error:
    state.update(status='FAIL',error=type(error).__name__+': '+str(error),wall_seconds=time.monotonic()-started)
    save();raise
