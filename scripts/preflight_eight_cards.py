"""Probe a whole eight-DCU allocation without executing model calculations.

--config-only is a CPU check of tensor-parallel partition sizes, not an eight-card
test. Full probing must run inside the allocated machine, never on a login node.
"""
import argparse
import faulthandler
import importlib.metadata
import json
import os
import shutil
import sys
from pathlib import Path


def check_partition(root):
    h3=json.loads((root/'transformer/config.json').read_text())
    qwen=json.loads((root/'text_encoder/config.json').read_text())
    text=qwen['text_config'];vision=qwen['vision_config']
    values={
        'h3_attention_heads':h3['num_attention_heads'],
        'h3_hidden_size':h3['hidden_size'],
        'h3_ffn_dim':h3['ffn_dim'],
        'qwen_attention_heads':text['num_attention_heads'],
        'qwen_kv_heads':text['num_key_value_heads'],
        'qwen_hidden_size':text['hidden_size'],
        'qwen_intermediate_size':text['intermediate_size'],
        'vision_attention_heads':vision['num_heads'],
        'vision_hidden_size':vision['hidden_size'],
        'vision_intermediate_size':vision['intermediate_size'],
    }
    return {name:dict(total=value,divisible_by_eight=value>0 and value%8==0,per_rank=value//8)
            for name,value in values.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-root',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--config-only',action='store_true')
    args=parser.parse_args()
    if args.report.exists():raise FileExistsError('Use a fresh report path')
    result=dict(partition_checks=check_partition(args.model_root),gpu_probe_executed=False,
                collective_operations_tested=False,native_multi_card_forward_tested=False,
                full_fl2va_tested=False,official_similarity_validated=False)
    checks={'partition_dimensions':all(v['divisible_by_eight'] for v in result['partition_checks'].values())}
    if not args.config_only:
        faulthandler.enable(all_threads=True)
        faulthandler.dump_traceback_later(60,repeat=True)
        result['versions']={}
        for name in ('torch','ninetoothed','triton','diffusers','transformers'):
            try:result['versions'][name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:result['versions'][name]=None
        try:
            print('EIGHT_CARD_PROBE_IMPORT_TORCH_BEGIN',flush=True)
            import torch
            print('EIGHT_CARD_PROBE_DEVICE_COUNT_BEGIN',flush=True)
            count=torch.cuda.device_count()
            print('EIGHT_CARD_PROBE_DEVICE_COUNT_PASS',count,flush=True)
            result.update(gpu_probe_executed=True,visible_devices=count,hip_version=torch.version.hip)
            checks.update(eight_visible_devices=count==8,hip_available=bool(torch.version.hip))
            devices=[]
            for index in range(count):
                print('EIGHT_CARD_PROBE_TORCH_PROPERTIES',index,flush=True)
                prop=torch.cuda.get_device_properties(index)
                with torch.cuda.device(index):free,total=torch.cuda.mem_get_info()
                devices.append(dict(index=index,name=prop.name,arch=getattr(prop,'gcnArchName',None),
                                    free_bytes=free,total_bytes=total))
            result['devices']=devices
            checks['matching_device_architectures']=bool(devices) and all(d['arch'] for d in devices) and len({d['arch'] for d in devices})==1
            result['peer_access']=[[i==j or bool(torch.cuda.can_device_access_peer(i,j)) for j in range(count)] for i in range(count)]
            # Missing peer access is diagnostic, not proof that collective fallback fails.
            import infinicore
            from infinilm.lib import _infinilm
            print('EIGHT_CARD_PROBE_NATIVE_IMPORT_PASS',flush=True)
            checks['native_nine_bridge']=all(hasattr(_infinilm,n) for n in ('_h3_to_dlpack','_h3_set_nine_callback'))
            checks['nine_compiler_packages_present']=all(result['versions'][n] for n in ('ninetoothed','triton'))
            # Some installed HIP runtimes omit gcnArchName in the legacy
            # property structure. Use the backend's supported target setting,
            # bound to the actual homogeneous Torch device identities.
            import triton
            backend=triton.runtime.driver.active
            print('EIGHT_CARD_PROBE_COMPILER_HELPER_BEGIN',flush=True)
            native_properties=[]
            for index in range(count):
                prop=torch.cuda.get_device_properties(index)
                native=backend.utils.get_device_properties(index)
                print('EIGHT_CARD_PROBE_COMPILER_PROPERTIES',index,native,flush=True)
                native_properties.append(native)
                torch_shared=getattr(prop,'shared_memory_per_block',None)
                checks[f'compiler_shared_memory_valid_device_{index}']=(
                    native['max_shared_mem']>0 and native['max_shared_mem']%1024==0)
                if torch_shared is not None:
                    checks[f'compiler_shared_memory_matches_torch_device_{index}']=(
                        native['max_shared_mem']==torch_shared)
                checks[f'compiler_warp_size_device_{index}']=(
                    native['warpSize']==getattr(prop,'warp_size',None))
                checks[f'compiler_multiprocessors_device_{index}']=(
                    native['multiprocessor_count']==prop.multi_processor_count)
            result['native_compiler_device_properties']=native_properties
            result['shared_memory_validation']=dict(source='typed HIP hipDeviceGetAttribute query',
                torch_comparison_available=hasattr(torch.cuda.get_device_properties(0),'shared_memory_per_block'),
                validity_checks='positive bytes aligned to 1024; exact Torch comparison when exposed')
            arch=devices[0]['arch'].split(':')[0]
            if not arch.startswith('gfx') or any(d['arch'].split(':')[0]!=arch for d in devices):
                raise RuntimeError('Cannot bind compiler target to actual homogeneous HIP devices')
            configured=os.environ.get('TRITON_FORCE_HIP_ARCH')
            if configured and configured!=arch:
                raise RuntimeError('Configured compiler architecture differs from actual allocation')
            os.environ['TRITON_FORCE_HIP_ARCH']=arch
            target=backend.get_current_target()
            result['compiler_target']=dict(arch=target.arch,backend=target.backend,warp_size=target.warp_size,
                selected_from='actual Torch gcnArchName for every allocated device',
                official_backend_setting='TRITON_FORCE_HIP_ARCH')
            checks['compiler_target_matches_actual_devices']=(target.arch==arch and target.backend=='hip')
        except Exception as error:
            result['environment_error']=type(error).__name__+': '+str(error)
            checks['runtime_import_and_device_queries']=False
        finally:
            faulthandler.cancel_dump_traceback_later()
    result['checks']=checks
    result['status']=('CONFIG_PARTITION_PRECHECK_PASS_NOT_DEVICE_VALIDATED' if args.config_only
                      else 'ENVIRONMENT_PROBE_PASS_NOT_MODEL_VALIDATION') if all(checks.values()) else 'ENVIRONMENT_NOT_READY'
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    return 0 if all(checks.values()) else 2


if __name__=='__main__':sys.exit(main())
