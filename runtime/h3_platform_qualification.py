"""Require recorded component evidence before inference on a new HIP target."""
import hashlib
import json
import os
from pathlib import Path
from h3_source_audit import inference_sources


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_hip_attention_platform(arch, torch_version, package=None):
    target = arch.split(':', 1)[0]
    if target == 'gfx906':
        return dict(status='LEGACY_GFX906_COMPONENT_QUALIFICATION', arch=arch)
    if target != 'gfx936':
        raise RuntimeError('HIP attention target has no qualification procedure: ' + arch)
    if os.environ.get('H3_PLATFORM_QUALIFICATION_MODE') == 'component-tests':
        return dict(status='UNQUALIFIED_COMPONENT_TESTS_ONLY', arch=arch)
    filename = os.environ.get('H3_PLATFORM_QUALIFICATION')
    if not filename:
        raise RuntimeError('gfx936 inference requires the completed platform qualification report')
    path = Path(filename).resolve()
    record = json.loads(path.read_text())
    if record.get('status') != 'PASS' or record.get('arch') != arch or record.get('torch') != torch_version:
        raise RuntimeError('Platform qualification does not match this HIP runtime')
    package = Path(package or Path(__file__).resolve().parents[1]).resolve()
    if record.get('inference_sources') != inference_sources(package):
        raise RuntimeError('Inference sources changed since platform qualification')
    if record.get('attention_backend_manifest_sha256') != _sha(package/'attention-backend-manifest.json'):
        raise RuntimeError('Attention backend changed since platform qualification')
    required = {'native_matrix', 'tiny_h3', 'tiny_qwen', 'pv_unit', 'pv_attention', 'block_softmax'}
    reports = record.get('test_reports', {})
    if set(reports) != required:
        raise RuntimeError('Incomplete platform component test evidence')
    for name, item in reports.items():
        report = Path(item['path'])
        if _sha(report) != item['sha256'] or json.loads(report.read_text()).get('status') != 'PASS':
            raise RuntimeError('Platform test evidence changed: ' + name)
    prefix = Path(os.environ['INFINI_ROOT']).resolve()
    libraries = record.get('native_libraries', {})
    required_libraries = {'libinfiniccl.so', 'libinfiniop.so', 'libinfinirt.so', 'libinfinicore_cpp_api.so'}
    if {Path(name).name for name in libraries} != required_libraries:
        raise RuntimeError('Native library qualification is missing')
    for name, expected in libraries.items():
        library = Path(name).resolve()
        if prefix not in library.parents or _sha(library) != expected:
            raise RuntimeError('Native libraries changed since platform qualification')
    compiler_files = record.get('compiler_files', {})
    if set(compiler_files) != {'triton_hip_driver', 'triton_hip_abi', 'nine_generation'}:
        raise RuntimeError('Compiler qualification is missing')
    for item in compiler_files.values():
        if _sha(item['path']) != item['sha256']:
            raise RuntimeError('Compiler changed since platform qualification')
    return dict(status='GFX936_COMPONENT_QUALIFICATION_PASS', arch=arch,
                report=str(path), report_sha256=_sha(path))
