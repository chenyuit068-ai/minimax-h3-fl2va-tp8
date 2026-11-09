#!/usr/bin/env python3
"""Compatible host wrapper around the supplied InfiniLM FL2VA entry."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
import time


def main():
    if os.environ.get('H3_PLATFORM_QUALIFICATION_MODE'):
        raise RuntimeError('Component qualification mode cannot run full-model generation')
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('task',choices=['fl2va'])
    p.add_argument('--model-path',required=True);p.add_argument('--first-frame',required=True)
    p.add_argument('--text',required=True);p.add_argument('--output',required=True)
    p.add_argument('--gpus',default='0');p.add_argument('--steps',type=int,default=50)
    p.add_argument('--num-frames',type=int,default=124);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--height',type=int);p.add_argument('--width',type=int)
    p.add_argument('--infinilm-root',type=Path,default=Path(os.environ.get('H3_ROOT','/root/private_data/h3-b0375'))/'Projects/InfiniLM')
    p.add_argument('--enable-native-fusions',action='store_true',help='Opt in to experimental native Nine fusions; validate the target first')
    p.add_argument('--experimental-streaming',action='store_true',help='Unvalidated full-model phase residency experiment; requires separate patched Core/LM')
    p.add_argument('--experimental-matrix-attention',action='store_true',help='Opt in to target-qualified Nine matrix attention; full video validation remains required')
    p.add_argument('--preflight-only',action='store_true')
    a=p.parse_args()
    if a.experimental_matrix_attention and not (a.experimental_streaming or a.enable_native_fusions):
        p.error('Matrix attention requires native fusions or experimental streaming')
    if not a.text.strip():p.error('Prompt must not be empty')
    if min(a.steps,a.num_frames)<1:p.error('Steps and frames must be positive')
    if (a.height is None)!=(a.width is None):p.error('Height and width must be supplied together')
    if a.height is not None and (min(a.height,a.width)<32 or a.height%32 or a.width%32):p.error('Dimensions must be positive multiples of 32')
    out=Path(a.output).resolve();out.parent.mkdir(parents=True,exist_ok=True)
    audit_path=out.with_suffix(out.suffix+'.run.json')
    # Keep every invocation's audit data. Do not reuse existing generated output.
    if out.exists() or audit_path.exists():raise FileExistsError(f'Choose a fresh output and audit path: {out}')
    from preflight import check
    audit=check(a.model_path,a.first_frame,gpus=a.gpus,
                memory_policy='experimental_streaming' if a.experimental_streaming else 'resident')
    audit['requested_configuration']={k:getattr(a,k) for k in ('task','text','gpus','steps','num_frames','seed','height','width')}
    audit['native_fusions_requested']=a.enable_native_fusions or a.experimental_streaming
    audit['experimental_streaming_requested']=a.experimental_streaming
    audit['experimental_matrix_attention_requested']=a.experimental_matrix_attention
    audit['invocation_id']=os.environ.get('H3_INVOCATION_ID')
    audit_path.write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    if audit['errors']:raise RuntimeError('Preflight blocked: '+'; '.join(audit['errors']))
    if a.preflight_only:print(json.dumps(audit));return
    entry=a.infinilm_root/'examples/minimax_h3.py'
    if not entry.is_file():raise FileNotFoundError(entry)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['NINETOOTHED_DISABLE_FP_FUSION']='1'
    sys.path.insert(0,str(Path(__file__).resolve().parent/'runtime'))
    args=['fl2va',f'--model-path={a.model_path}',f'--first-frame={a.first_frame}',f'--text={a.text}',
          f'--output={out}',f'--gpus={a.gpus}',f'--steps={a.steps}',f'--num-frames={a.num_frames}',f'--seed={a.seed}']
    if a.height is not None:args.extend([f'--height={a.height}',f'--width={a.width}'])
    old_argv=sys.argv;runtime=None;phases=None;started=time.perf_counter()
    audit['status']='RUNNING';audit['full_fl2va_tested']=False
    # Persist before entering native code; the waiting parent must see RUNNING
    # even if a native signal prevents this process from reaching finally.
    audit_path.write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    try:
        from h3_source_audit import capture
        audit['source_identity']=capture(Path(__file__).resolve().parent,entry)
        audit_path.write_text(json.dumps(audit,ensure_ascii=False,indent=2))
        if a.enable_native_fusions or a.experimental_streaming:
            from h3_nine_runtime import NativeFusions
            runtime=NativeFusions(attention=True,matrix_attention=a.experimental_matrix_attention)
        sys.argv=[str(entry),*args]
        if a.experimental_streaming:
            spec=importlib.util.spec_from_file_location('h3_streaming_reference_entry',entry)
            module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
            from h3_phase_pipeline import make_pipeline
            def load_streaming(paths,task,devices,timings,**options):
                nonlocal phases
                if task!='fl2va' or options.get('legacy'):raise ValueError('Only native FL2VA supports this experiment')
                pipe,text,transformer=make_pipeline(module,paths,devices,timings,runtime=runtime)
                phases=pipe._h3_phase_residency
                return pipe,text,transformer
            module.load_pipeline=load_streaming
            module.main()
        else:runpy.run_path(str(entry),run_name='__main__')
        import torch
        for device in audit['devices']:torch.cuda.synchronize(device['index'])
        if not out.is_file() or out.stat().st_size==0:raise RuntimeError('Entry returned without a video')
        audit['status']='GENERATION_COMPLETED_UNVALIDATED'
        audit['full_fl2va_tested']=True
    except BaseException as exc:
        audit['status']='FAILED';audit['error']=f'{type(exc).__name__}: {exc}';raise
    finally:
        sys.argv=old_argv
        if phases is not None:
            audit['phase_transitions']=phases.transitions
            try:phases.release()
            except Exception as exc:audit['phase_cleanup_error']=f'{type(exc).__name__}: {exc}'
        if runtime is not None:
            audit['platform_qualification']=runtime.platform_qualification
            audit['fusion_calls']=dict(runtime.calls)
            audit['attention_routes']=dict(runtime.attention_routes)
            audit['peak_nine_workspace_bytes']=runtime.peak_workspace_bytes
            audit['nine_workspace_evictions']=runtime.workspace_evictions
            runtime.close()
        audit['wall_seconds_including_startup']=time.perf_counter()-started
        audit['competition_passed']=False # Requires independent reference/spec validation.
        audit_path.write_text(json.dumps(audit,ensure_ascii=False,indent=2))

if __name__=='__main__':
    if os.environ.get('H3_WRAPPER_CHILD')=='1':
        import faulthandler
        faulthandler.enable(all_threads=True)
        main()
    else:
        # A Python finally block cannot catch SIGSEGV in a native extension.
        # The parent waits for all work and updates only this invocation's audit.
        probe=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
        probe.add_argument('--output');known,_=probe.parse_known_args()
        sys.path.insert(0,str(Path(__file__).resolve().parent/'runtime'))
        from h3_process_supervisor import supervise
        audit_path=Path(known.output).resolve().with_suffix(Path(known.output).suffix+'.run.json') if known.output else None
        raise SystemExit(supervise([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]],audit_path))
