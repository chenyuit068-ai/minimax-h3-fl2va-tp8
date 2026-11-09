#!/usr/bin/env python3
"""Eight-device resident FL2VA adapter; delegates generation to the reference entry.

One host process uses InfiniLM's native TP worker threads. Do not launch eight
copies of this script with torchrun. All new numeric kernels remain NineToothed.
"""
import argparse
import functools
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE/'runtime'))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=['fl2va'])
    parser.add_argument('--infinilm-root', type=Path, required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--first-frame', required=True)
    parser.add_argument('--text', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--num-frames', type=int, default=124)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--height', type=int)
    parser.add_argument('--width', type=int)
    parser.add_argument('--workspace-mib-per-device', type=int, default=1536)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    try:devices = [int(value) for value in args.gpus.split(',')]
    except ValueError:parser.error('--gpus must contain integer device indices')
    if sorted(devices) != list(range(8)):
        parser.error('An allocated whole eight-device machine is required: --gpus 0,1,2,3,4,5,6,7')
    if not args.text.strip() or min(args.steps, args.num_frames, args.workspace_mib_per_device) < 1:
        parser.error('Prompt and positive generation/workspace parameters are required')
    if (args.height is None) != (args.width is None):
        parser.error('--height and --width must be supplied together')
    if args.height is not None and (min(args.height,args.width)<32 or args.height%32 or args.width%32):
        parser.error('Dimensions must be positive multiples of 32')
    return args, devices


def main():
    args, devices = parse_args()
    if os.environ.get('H3_PLATFORM_QUALIFICATION_MODE'):
        raise RuntimeError('Component-test mode cannot perform full-model generation')
    if os.environ.get('INFINICORE_DEFER_PARAMETER_STORAGE'):
        raise RuntimeError('Resident TP8 must not inherit deferred single-device weight storage')
    out = Path(args.output).expanduser().resolve()
    audit_path = out.with_suffix(out.suffix+'.run.json')
    if out.exists() or audit_path.exists():
        raise FileExistsError('Choose a fresh output/audit path; existing results are never reused')
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    from preflight import check
    audit = check(args.model_path, args.first_frame, gpus=args.gpus, memory_policy='resident')
    audit.update(status='PREFLIGHT', invocation_id=os.environ.get('H3_INVOCATION_ID'),
                 execution_mode='native_resident_tp8', official_similarity_validated=False,
                 competition_passed=False, requested_configuration=vars(args).copy())
    audit['requested_configuration']['infinilm_root'] = str(args.infinilm_root.resolve())
    def save():audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    save()
    try:
        if audit['errors']:raise RuntimeError('Preflight blocked: '+'; '.join(audit['errors']))
        import torch
        if torch.cuda.device_count()!=8 or not torch.version.hip:
            raise RuntimeError('Exactly eight allocated HIP devices must be visible')
        if args.preflight_only:
            audit['status']='PREFLIGHT_ONLY_PASS_NOT_GENERATION';save();return
        entry=args.infinilm_root.resolve()/'examples/minimax_h3.py'
        if not entry.is_file():raise FileNotFoundError(entry)
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',NINETOOTHED_DISABLE_FP_FUSION='1')
        from h3_source_audit import capture
        from h3_eightcard_evidence import sha, validate_eightcard_report, validate_original_model_view
        audit['source_identity']=capture(PACKAGE,entry)
        audit['component_qualification']=validate_eightcard_report(PACKAGE)
        audit['original_model_view']=validate_original_model_view(args.model_path,PACKAGE)
        audit['first_frame_sha256']=sha(args.first_frame)
    except BaseException as error:
        audit.update(status='FAILED_PRECONDITION',error=type(error).__name__+': '+str(error));save();raise
    from h3_framework_streams import TorchNativeHandoff
    with TorchNativeHandoff() as handoff:
        runtime=None;module=None;vae=None;original_decode=None;original_argv=sys.argv;resident_components=[]
        audit['status']='RUNNING';save()
        try:
            from h3_nine_runtime import NativeFusions
            with torch.cuda.device(devices[0]):
                runtime=NativeFusions(attention=True,matrix_attention=True,device_ids=devices,
                    cache_policy='pinned',max_signatures=64,
                    max_workspace_bytes=args.workspace_mib_per_device*1024**2)
            spec=importlib.util.spec_from_file_location('h3_tp8_reference_entry',entry)
            module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
            original_load=module.load_pipeline
            def load_resident(paths,task,selected,timings,**options):
                nonlocal vae,original_decode
                if task!='fl2va' or options.get('legacy') or selected!=devices:
                    raise ValueError('This adapter supports native whole-machine FL2VA only')
                result=original_load(paths,task,selected,timings,**options)
                # Keep native engines alive until their cached stream-owned scratch
                # has been synchronized and released in the outer finally block.
                resident_components.extend(result)
                vae=result[0].vae;original_decode=vae.decode
                @functools.wraps(original_decode)
                def decode(*positional,**keywords):
                    from h3_vae_attention import VideoVAEAttention
                    with torch.cuda.device(vae.device),VideoVAEAttention() as attention:
                        answer=original_decode(*positional,**keywords)
                    audit['video_vae_nine']=dict(calls=attention.calls,peak_workspace_bytes=attention.peak_workspace_bytes)
                    save();return answer
                vae.decode=decode
                return result
            module.load_pipeline=load_resident
            argv=['fl2va',f'--model-path={args.model_path}',f'--first-frame={args.first_frame}',
                  f'--text={args.text}',f'--output={out}',f'--gpus={args.gpus}',
                  f'--steps={args.steps}',f'--num-frames={args.num_frames}',f'--seed={args.seed}']
            if args.height is not None:argv.extend([f'--height={args.height}',f'--width={args.width}'])
            sys.argv=[str(entry),*argv]
            module.main()
            for index in devices:torch.cuda.synchronize(index)
            if not out.is_file() or not out.stat().st_size:
                raise RuntimeError('Reference entry returned without a nonempty video')
            observed=set(runtime.calls_by_device)
            if observed!=set(devices) or any(runtime.calls_by_device[i]['attention']==0 for i in devices):
                raise RuntimeError('Generation did not exercise Nine attention on all eight TP devices')
            audit.update(status='GENERATION_COMPLETED_UNVALIDATED',full_fl2va_tested=True,
                         video_sha256=sha(out),video_bytes=out.stat().st_size)
        except BaseException as error:
            audit.update(status='FAILED',error=type(error).__name__+': '+str(error),full_fl2va_tested=False)
            raise
        finally:
            audit['framework_handoff']=handoff.snapshot()
            sys.argv=original_argv
            if vae is not None and original_decode is not None:vae.decode=original_decode
            if runtime is not None:
                audit.update(fusion_calls=dict(runtime.calls),
                    fusion_calls_by_device={str(i):dict(v) for i,v in runtime.calls_by_device.items()},
                    peak_nine_workspace_bytes_all_devices=runtime.peak_workspace_bytes,
                    workspace_limit_bytes_per_device=runtime.max_workspace_bytes,
                    nine_workspace_evictions=runtime.workspace_evictions)
                try:runtime.close()
                except Exception as error:
                    audit.update(status='FAILED_CLEANUP',cleanup_error=type(error).__name__+': '+str(error))
                    save();raise
            audit['wall_seconds_including_startup']=time.perf_counter()-started
            save()
            resident_components.clear()


if __name__=='__main__':
    if os.environ.get('H3_WRAPPER_CHILD')=='1':
        import faulthandler
        faulthandler.enable(all_threads=True)
        main()
    else:
        from h3_process_supervisor import supervise
        probe=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
        probe.add_argument('--output');known,_=probe.parse_known_args()
        path=Path(known.output).expanduser().resolve() if known.output else None
        audit=path.with_suffix(path.suffix+'.run.json') if path else None
        raise SystemExit(supervise([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]],audit))
