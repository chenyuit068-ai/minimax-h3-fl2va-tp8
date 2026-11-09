"""Bind eight-device component evidence to the actual source and runtime files."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

from h3_source_audit import inference_sources

REQUIRED_TESTS = {'collectives', 'native_eight_devices', 'tiny_tp8_h3', 'tiny_tp8_qwen'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def runtime_identity(package):
    import torch
    import triton
    import ninetoothed
    from infinilm.lib import _infinilm

    if torch.cuda.device_count() != 8 or not torch.version.hip:
        raise RuntimeError('This qualification requires exactly eight allocated HIP devices')
    prefix = Path(os.environ['INFINI_ROOT']).resolve()
    files = [prefix/'lib'/name for name in (
        'libinfiniccl.so', 'libinfiniop.so', 'libinfinirt.so', 'libinfinicore_cpp_api.so')]
    files.extend([Path(_infinilm.__file__), Path(ninetoothed.__file__).parent/'generation.py'])
    backends = Path(triton.__file__).parent/'backends'
    driver = next((p for p in (backends/'hcu', backends/'amd')
                   if (p/'driver.py').is_file()), None)
    if driver is None:
        raise RuntimeError('Cannot identify the installed HIP compiler driver')
    files.extend([driver/'driver.py', driver/'driver.c', Path(triton.__file__).parent/'runtime/cache.py'])
    # Include the Python facades and weight sharder used by native TP loading.
    import infinilm.modeling_utils as loading
    import infinilm.multimodal.minimax_h3 as model
    files.extend([Path(loading.__file__), Path(model.__file__)])
    versions = {}
    for name in ('torch', 'ninetoothed', 'triton', 'flagtree', 'diffusers', 'transformers'):
        try:versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name] = None
    devices = []
    for index in range(8):
        prop = torch.cuda.get_device_properties(index)
        devices.append(dict(index=index, name=prop.name, arch=prop.gcnArchName,
                            total_bytes=prop.total_memory))
    if len({d['arch'] for d in devices}) != 1:
        raise RuntimeError('Mixed device architectures require separate qualification')
    return dict(inference_sources=inference_sources(package),
                attention_backend_manifest_sha256=sha(Path(package)/'attention-backend-manifest.json'),torch=torch.__version__,
                hip=torch.version.hip, python=sys.version, versions=versions, devices=devices,
                compiler_target=str(triton.runtime.driver.active.get_current_target()),
                compiler_arch_setting=os.environ.get('TRITON_FORCE_HIP_ARCH'),
                compiler_scalar_properties_setting=os.environ.get('TRITON_HIP_SCALAR_PROPERTIES'),
                staged_model_dependency_wheels=json.loads(os.environ.get('H3_MODEL_DEPENDENCY_WHEELS_JSON','{}')),
                environment_files={str(p.resolve()):sha(p) for p in files})


def validate_eightcard_report(package=None):
    package = Path(package or Path(__file__).resolve().parents[1]).resolve()
    filename = os.environ.get('H3_EIGHT_CARD_QUALIFICATION')
    if not filename:
        raise RuntimeError('Set H3_EIGHT_CARD_QUALIFICATION to completed eight-device component evidence')
    path = Path(filename).resolve()
    record = json.loads(path.read_text())
    if record.get('status') != 'EIGHT_CARD_COMPONENTS_PASS':
        raise RuntimeError('Eight-device component qualification has not passed')
    if record.get('runtime_identity') != runtime_identity(package):
        raise RuntimeError('Eight-device source, compiler, model facade or runtime changed after qualification')
    reports = record.get('test_reports', {})
    if set(reports) != REQUIRED_TESTS:
        raise RuntimeError('Incomplete eight-device component evidence')
    for name, item in reports.items():
        report = Path(item['path'])
        if sha(report) != item['sha256']:
            raise RuntimeError('Changed component evidence: '+name)
        evidence = json.loads(report.read_text())
        if evidence.get('status') != 'PASS' or evidence.get('world_size') != 8:
            raise RuntimeError('Failed or non-eight-device evidence: '+name)
    return dict(status='EIGHT_CARD_COMPONENTS_PASS', report=str(path), report_sha256=sha(path),
                full_fl2va_tested=False, official_similarity_validated=False)


def validate_original_model_view(model_path,package=None):
    # prepare_model_view.py performs the expensive full hash scan once on the
    # CPU login node and creates only links. This check binds that manifest to
    # the expected original files, without copying 144 GB into personal AFS.
    from preflight import fl2va_model_directory
    package=Path(package or Path(__file__).resolve().parents[1]).resolve()
    root=fl2va_model_directory(model_path)
    expected=json.loads((package/'assets/original-model-files.json').read_text())
    path=root/'asset-download-manifest.json';record=json.loads(path.read_text())
    if not record.get('complete') or any(record.get(name)!=expected[name] for name in ('repo','revision')):
        raise RuntimeError('A fully verified original model view is required')
    rows=record.get('files',[]);verified={row['file']:row for row in rows}
    if len(rows)!=len(verified) or set(verified)!={row['file'] for row in expected['files']}:
        raise RuntimeError('Original asset verification manifest is incomplete or ambiguous')
    for row in expected['files']:
        actual=verified[row['file']];target=root/row['file']
        if (actual.get('sha256')!=row['sha256'] or actual.get('verified_sha256')!=row['sha256'] or
                actual.get('bytes')!=row['bytes'] or target.stat().st_size!=row['bytes']):
            raise RuntimeError('Original model view identity differs: '+row['file'])
    return dict(status='ORIGINAL_ASSET_VERIFICATION_MANIFEST_MATCHES',repo=record['repo'],
                revision=record['revision'],files=len(verified),manifest_sha256=sha(path),
                full_hash_scan_repeated_during_generation=False,weight_conversion=False)
