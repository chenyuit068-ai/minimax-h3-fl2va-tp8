"""Collect only named H3 experiment outputs and executed source, never credentials."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import zipfile


def collect(root, output):
    if output.exists():
        raise FileExistsError('Use a fresh evidence archive name')
    package = root/'native-candidate-20261019/submission'
    repo = root/'participant-native/InfiniLM'
    final = root/'native-results-20261019-v8'
    correctness = json.loads((final/'native-dispatch.json').read_text())
    benchmark = json.loads((final/'native-benchmark.json').read_text())
    host = json.loads((final/'host-tools.json').read_text())
    assert all(row['status'] == 'PASS' for row in (correctness, benchmark, host))
    assert 'ADAPTIVE_COMPONENT_EXIT=2' in (root/'logs/native-v8.log').read_text()
    files = {}
    for folder in sorted(root.glob('native-results-20261019-v*')):
        for path in folder.rglob('*'):
            if path.is_file() and path.suffix in ('.json', '.log', '.exit'):
                files[path.relative_to(root).as_posix()] = path.read_bytes()
    for name in ('native-integration-build.log', 'native-integration-build.log.exit',
                 'native-checks-launch.log', 'native-checks-launch.log.exit',
                 *[f'native-v{i}.log' for i in range(2, 9)]):
        path = root/'logs'/name
        if path.is_file():
            files['logs/'+name] = path.read_bytes()
    for path in package.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix in (
                '.py', '.hpp', '.h', '.sh', '.json', '.patch'):
            files['executed-source/'+path.relative_to(package).as_posix()] = path.read_bytes()
    for name in ('csrc/models/minimax_h3/minimax_h3_transformer.cpp',
                 'csrc/pybind11/bindings.cc',
                 *['csrc/models/minimax_h3/'+n for n in (
                     'h3_nine_bridge.hpp', 'h3_nine_bindings.hpp', 'h3_dlpack.h')]):
        files['native-audit/'+name] = (repo/name).read_bytes()
    versions = {'python': platform.python_version()}
    for name in ('torch', 'ninetoothed', 'triton', 'pybind11'):
        versions[name] = importlib.metadata.version(name)
    files['software-versions.json'] = json.dumps(versions, indent=2).encode()
    hashes = [dict(file=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
              for name, data in sorted(files.items())]
    files['execution-hashes.json'] = json.dumps(hashes, indent=2).encode()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            archive.writestr(name, data)
    print(json.dumps(dict(file=str(output), bytes=output.stat().st_size,
        sha256=hashlib.file_digest(output.open('rb'), 'sha256').hexdigest(),
        output_checks=correctness['total_output_checks'], benchmark_cases=len(benchmark['cases']),
        host_checks=len(host['checks']), full_fl2va_tested=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    collect(args.root.resolve(), args.output.resolve())
