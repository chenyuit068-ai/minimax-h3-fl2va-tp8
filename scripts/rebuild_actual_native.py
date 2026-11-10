"""Reconstruct pinned actual-build sources and build in a fresh Linux directory.

This command never requests accelerators or runs model inference. Use the
competition's existing vendor Torch and DTK; it does not replace either one.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile

PACKAGE = Path(__file__).resolve().parents[1]
PINS = {
    'h3-source-snapshots.tar.gz': 'cc33835e8d7e48410fb0ab6098905e3fe6494b52dc9ce13244f938d5ac808890',
    'xmake-v3.0.5.tar.gz': 'b947666281222f79e082283b6f84e68880c499305890f6ab8b03b8bac82456dc',
    'xmake-repo-offline.tar.gz': 'c4c25e202ce1236d51e4478e0fa08e4148505c474f403812912516557df81ab2',
    'boost-build-prerequisites-20261107-r1.zip': '2515b43db92e4da760abf1a054dbe337aa5ef231f3b1f4db61cfa6f67a3bc8c5',
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def deb_data(blob):
    if blob[:8] != b'!<arch>\n':
        raise ValueError('Invalid Debian archive')
    offset = 8
    members = []
    while offset < len(blob):
        header = blob[offset:offset + 60]
        if len(header) != 60 or header[58:60] != b'`\n':
            raise ValueError('Invalid ar member')
        length = int(header[48:58])
        name = header[:16].decode().strip().rstrip('/')
        start = offset + 60
        payload = blob[start:start + length]
        if len(payload) != length:
            raise ValueError('Truncated Debian archive')
        if name.startswith('data.tar'):
            members.append((name, payload))
        offset = start + length + length % 2
    if len(members) != 1:
        raise ValueError('Expected one Debian data member')
    name, payload = members[0]
    if name.endswith('.zst'):
        library = ctypes.CDLL('libzstd.so.1')
        library.ZSTD_getFrameContentSize.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        library.ZSTD_getFrameContentSize.restype = ctypes.c_ulonglong
        library.ZSTD_decompress.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
        library.ZSTD_decompress.restype = ctypes.c_size_t
        library.ZSTD_isError.argtypes = [ctypes.c_size_t]
        library.ZSTD_isError.restype = ctypes.c_uint
        size = library.ZSTD_getFrameContentSize(payload, len(payload))
        if size >= 2 ** 64 - 2:
            size = 256 * 1024 ** 2
        if not 0 < size <= 256 * 1024 ** 2:
            raise ValueError('Unexpected Debian decompression size')
        buffer = ctypes.create_string_buffer(size)
        actual = library.ZSTD_decompress(buffer, size, payload, len(payload))
        if library.ZSTD_isError(actual):
            raise ValueError('Debian payload decompression failed')
        payload = buffer.raw[:actual]
    return payload


def install_xmake_without_metadata(source, prefix):
    """Install this build without copying ownership or ACL metadata."""
    binary = source / 'build/xmake'
    destination = prefix / 'bin/xmake'
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Some competition mounts permit file creation but reject cp -p's later
    # metadata restoration. Set executable mode at creation instead.
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o755)
    with binary.open('rb') as src, os.fdopen(descriptor, 'wb') as dst:
        shutil.copyfileobj(src, dst)
    program = prefix / 'share/xmake'
    program.mkdir(parents=True, exist_ok=False)
    count = 1
    for path in sorted((source / 'xmake').rglob('*')):
        target = program / path.relative_to(source / 'xmake')
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            count += 1
    if not os.access(destination, os.X_OK):
        raise RuntimeError('Fresh xmake is not executable on this filesystem')
    return dict(binary_sha256=sha(binary), installed_sha256=sha(destination),
                files_copied=count, metadata_preserved=False)


def prepare_compiler_directories(project, target):
    """Create source parent directories before parallel compiler workers start."""
    relative_directories = set()
    for source_root in ('src', 'csrc', 'python'):
        directory = project / source_root
        if not directory.is_dir():
            continue
        for path in directory.rglob('*'):
            if path.is_file() and path.suffix in ('.c', '.cc', '.cpp', '.cxx', '.cu'):
                relative_directories.add(path.parent.relative_to(project))
    for kind in ('.objs', '.deps'):
        output = project / 'build' / kind / target / 'linux/x86_64/release'
        for relative in sorted(relative_directories):
            (output / relative).mkdir(parents=True, exist_ok=True)
    return len(relative_directories) * 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-root', type=Path, required=True, help='New writable local scratch directory')
    parser.add_argument('--dtk-root', type=Path)
    parser.add_argument('--xmake', type=Path, help='Existing xmake 3.0.5; otherwise build the pinned source')
    parser.add_argument('--jobs', type=int, default=1,
                        help='Compiler processes; use 1 on mounts with concurrent mkdir races')
    parser.add_argument('--prepare-only', action='store_true', help='Validate and unpack sources and build dependencies only')
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('Use the competition Linux environment or Linux for host-only preparation')
    if not 1 <= args.jobs <= 16:
        parser.error('--jobs must be between 1 and 16')
    root = args.build_root.resolve()
    if root.exists():
        raise FileExistsError('Choose a fresh build-root; existing build evidence is preserved')
    prerequisites = PACKAGE / 'prerequisites'
    for name, expected in PINS.items():
        if sha(prerequisites / name) != expected:
            raise ValueError('Offline dependency identity differs: ' + name)
    root.mkdir(parents=True)
    report_path = root / 'build-report.json'
    report = dict(status='PREPARING',started_utc=datetime.now(timezone.utc).isoformat(),
                  gpu_operations_executed=False,model_inference_executed=False,phases=[],inputs=PINS)
    env = dict(os.environ, XMAKE_ROOT='y', PYTHONNOUSERSITE='1',
               H3_BUILD_DEVICE_IMPORT_CHECK='0', PYTORCH_NVML_BASED_CUDA_CHECK='1')

    def save():
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')

    def run(name, command, cwd=None, timeout=7200):
        print('NATIVE_BUILD_PHASE', name, flush=True)
        log = root / (name + '.log')
        started = time.monotonic()
        with log.open('w') as stream:
            result = subprocess.run([str(x) for x in command], cwd=cwd, env=env,
                                    stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
        report['phases'].append(dict(name=name,returncode=result.returncode,
                                     seconds=time.monotonic()-started,log=log.name,sha256=sha(log)))
        save()
        if result.returncode:
            raise RuntimeError(name + ' failed; inspect ' + str(log))

    save()
    try:
        run('reconstruct_sources', [sys.executable, PACKAGE / 'scripts/assemble_actual_native_sources.py',
             '--manifest', PACKAGE / 'reproduction/native-source-overlay-manifest.json',
             '--source-archive', prerequisites / 'h3-source-snapshots.tar.gz', '--output', root / 'native'])
        sysroot = root / 'sysroot'
        sysroot.mkdir()
        with zipfile.ZipFile(prerequisites / 'boost-build-prerequisites-20261107-r1.zip') as archive:
            rows = json.loads(archive.read('build-prerequisites-manifest.json'))['files']
            for row in rows:
                raw = archive.read(row['file'])
                if len(raw) != row['bytes'] or hashlib.sha256(raw).hexdigest() != row['sha256']:
                    raise ValueError('Boost package identity differs')
                with tarfile.open(fileobj=io.BytesIO(deb_data(raw)), mode='r:*') as payload:
                    payload.extractall(sysroot, filter='data')
        repository = root / 'cache/xmake/.xmake/repositories'
        repository.mkdir(parents=True)
        with tarfile.open(prerequisites / 'xmake-repo-offline.tar.gz') as archive:
            archive.extractall(repository, filter='data')
        fetch = repository / 'xmake-repo/packages/b/boost/fetch.lua'
        text = fetch.read_text()
        needle = '        local result = {\n            libfiles = {},'
        replacement = '''        local h3_boost_root = os.getenv("H3_BOOST_ROOT")
        if h3_boost_root then
            table.insert(paths, 1, path.join(h3_boost_root, "usr", "lib", "x86_64-linux-gnu"))
        end
        local result = {
            includedirs = h3_boost_root and {path.join(h3_boost_root, "usr", "include")} or {},
            libfiles = {},'''
        if text.count(needle) != 1:
            raise ValueError('Pinned offline Boost lookup source differs')
        fetch.write_text(text.replace(needle, replacement, 1))
        report['offline_boost_lookup_sha256'] = sha(fetch)
        report['source_reconstruction_sha256'] = sha(root / 'native/source-reconstruction.json')
        report['status'] = 'OFFLINE_SOURCE_AND_BUILD_DEPENDENCIES_PREPARED_NOT_A_NATIVE_BUILD'
        save()
        if args.prepare_only:
            print(json.dumps(report, indent=2));return
        if args.dtk_root is None:
            raise ValueError('--dtk-root is required for native compilation')
        dtk = args.dtk_root.resolve()
        if not (dtk / 'cuda/cuda/bin/nvcc').is_file() or not (dtk / 'include/hip/hip_runtime.h').is_file():
            raise ValueError('Matching DTK CUDA compatibility compiler and HIP headers are required')
        if shutil.disk_usage(root).free < 2 * 1024 ** 3:
            raise RuntimeError('Compilation requires at least 2 GiB of remaining local scratch')
        python = Path(sys.executable)
        dependency_probe = subprocess.run([str(python),'-c',
            'import importlib.metadata as m,json,torch,sysconfig;'
            'assert m.version("torch")=="2.9.0+das.opt1.dtk2604";'
            'assert m.version("pybind11")=="2.13.6";'
            'print(json.dumps({"abi":int(torch._C._GLIBCXX_USE_CXX11_ABI),'
            '"torch_lib":str(__import__("pathlib").Path(torch.__file__).parent/"lib"),'
            '"site":sysconfig.get_paths()["purelib"]}))'],env=env,capture_output=True,text=True,timeout=60,check=True)
        probe = json.loads(dependency_probe.stdout.strip().splitlines()[-1])
        prefix = root / 'infini'
        prefix.mkdir()
        tools = root / 'tools'
        env.update(PYTHON=str(python),INFINI_ROOT=str(prefix),H3_BOOST_ROOT=str(sysroot),
                   DTK_ROOT=str(dtk),HYGON_ARCH='gfx936',CC='gcc',CXX='g++',TMPDIR=str(root),
                   XMAKE_GLOBALDIR=str(root/'cache/xmake'),INFINILM_CXX11_ABI=str(probe['abi']),
                   CPLUS_INCLUDE_PATH=str(sysroot/'usr/include'),LIBRARY_PATH=str(sysroot/'usr/lib/x86_64-linux-gnu'),
                   PKG_CONFIG_PATH=probe['site']+'/pybind11/share/pkgconfig',
                   CMAKE_PREFIX_PATH=probe['site']+'/pybind11/share/cmake')
        if args.xmake:
            xmake = args.xmake.resolve()
            program = xmake.parent.parent / 'share/xmake'
            if program.is_dir():
                env['XMAKE_PROGRAM_DIR'] = str(program)
        else:
            tool_source = root / 'tool-source'
            tool_source.mkdir()
            with tarfile.open(prerequisites/'xmake-v3.0.5.tar.gz') as archive:
                archive.extractall(tool_source,filter='data')
            xmake_source = tool_source/'xmake-3.0.5'
            run('configure_xmake',['bash','./configure','--prefix='+str(tools)],cwd=xmake_source)
            run('compile_xmake',['make','-j'+str(args.jobs)],cwd=xmake_source)
            started = time.monotonic()
            installation = install_xmake_without_metadata(xmake_source, tools)
            report['phases'].append(dict(name='install_xmake_without_metadata',
                returncode=0, seconds=time.monotonic()-started, **installation))
            save()
            xmake = tools/'bin/xmake'
            env['XMAKE_PROGRAM_DIR']=str(tools/'share/xmake')
        version = subprocess.run([str(xmake),'--version'],env=env,capture_output=True,text=True,check=True,timeout=30)
        if '3.0.5' not in version.stdout:
            raise RuntimeError('Use the pinned xmake 3.0.5')
        env['PATH']=':'.join(map(str,[xmake.parent,python.parent,dtk/'cuda/cuda/bin',dtk/'llvm/bin']))+':'+env['PATH']
        env['LD_LIBRARY_PATH']=':'.join(map(str,[prefix/'lib',sysroot/'usr/lib/x86_64-linux-gnu',
            probe['torch_lib'],dtk/'cuda/cuda/lib64',dtk/'lib',dtk/'lib64']))+':'+env.get('LD_LIBRARY_PATH','')
        core=root/'native/source/Projects/InfiniCore'
        lm=root/'native/source/Projects/InfiniLM'
        run('xmake_private_network',[xmake,'global','--network=private'])
        run('configure_core',[xmake,'f','-y','-cv','--mode=release','--toolchain=gcc','--omp=n',
            '--hygon-dcu=y','--hygon-arch=gfx936','--cuda='+str(dtk/'cuda/cuda'),'--ccl=y','--aten=y'],cwd=core)
        run('configure_lm',[xmake,'f','-y','-cv','--mode=release'],cwd=lm)
        for target in ('infinirt','infiniop','infiniccl','infinicore_cpp_api','_infinicore'):
            prepared = prepare_compiler_directories(core, target)
            report.setdefault('prepared_compiler_directories', {})[target] = prepared
            save()
            run('build_'+target,[xmake,'build','-j'+str(args.jobs),target],cwd=core)
            run('install_'+target,[xmake,'install',target],cwd=core)
        report['prepared_compiler_directories']['_infinilm'] = prepare_compiler_directories(lm, '_infinilm')
        save()
        run('build_infinilm',[xmake,'build','-j'+str(args.jobs),'_infinilm'],cwd=lm)
        run('install_infinilm',[xmake,'install','_infinilm'],cwd=lm)
        for library in (sysroot/'usr/lib/x86_64-linux-gnu').glob('libboost*.so*'):
            if library.is_file():
                shutil.copyfile(library,prefix/'lib'/library.name)
        report.update(status='NATIVE_COMPILATION_PASS_DEVICE_QUALIFICATION_REQUIRED',
                      prefix=str(prefix),torch_cxx11_abi=probe['abi'],completed_utc=datetime.now(timezone.utc).isoformat())
        save();print(json.dumps(report,indent=2))
    except BaseException as error:
        report.update(status='FAIL_NO_AUTOMATIC_RETRY',error=type(error).__name__+': '+str(error))
        save();raise


if __name__ == '__main__':
    main()
