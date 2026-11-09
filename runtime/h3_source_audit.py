"""Record inference source bytes; contains no model inputs or credentials."""
import hashlib
import json
from pathlib import Path


def inference_sources(package):
    package = Path(package).resolve()
    paths = [package/'run_fl2va.py', package/'preflight.py']
    if (package/'assets/original-model-files.json').is_file():paths.append(package/'assets/original-model-files.json')
    if (package/'run_eightcard.py').is_file():paths.append(package/'run_eightcard.py')
    paths.extend((package/'runtime').glob('*.py'))
    for directory in ('native', 'streaming'):
        paths.extend(p for p in (package/directory).rglob('*')
                     if p.is_file() and p.suffix in ('.h', '.hpp', '.cpp', '.cc', '.patch', '.json'))
    return {p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(paths))}


def capture(package, reference_entry):
    package = Path(package).resolve()
    result = dict(schema_version=1, inference_sources=inference_sources(package),
                  reference_entry_sha256=hashlib.sha256(Path(reference_entry).read_bytes()).hexdigest())
    backend = package/'attention-backend-manifest.json'
    if backend.is_file():
        manifest = json.loads(backend.read_text())
        for name, expected in manifest['runtime_sources'].items():
            if Path(name).name != name:
                raise ValueError('Invalid declared runtime source name')
            actual = result['inference_sources'].get('runtime/'+name)
            if actual != expected:
                raise ValueError('Selected attention source changed: '+name)
        result.update(attention_backend=manifest['selected_backend'],
                      attention_backend_manifest_sha256=hashlib.sha256(backend.read_bytes()).hexdigest())
    return result
