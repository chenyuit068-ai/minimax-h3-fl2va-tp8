"""Copy verified original Python wheels into local scratch; no model math."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

PINS = {
    'transformers': ('5.19.0', 'afcd2dd5f603ed28c1e1fcb00a338ccbb4ef5f878ed289635df8b58187afb518'),
    'diffusers': ('0.41.0', 'ea8918b7dfd92ce793b6db689a551b8cd25d52c6c0fcd3ade0706a5fd2a25990'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheelhouse', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError('Choose fresh scratch; existing files are preserved')
    verified = {}
    for name, (version, expected) in PINS.items():
        wheel = args.wheelhouse / f'{name}-{version}-py3-none-any.whl'
        if hashlib.sha256(wheel.read_bytes()).hexdigest() != expected:
            raise ValueError('Pinned wheel differs: ' + wheel.name)
        verified[name] = wheel
    output.mkdir(parents=True)
    count = 0
    for name, wheel in verified.items():
        with zipfile.ZipFile(wheel) as archive:
            if archive.testzip() is not None:
                raise ValueError('Wheel CRC verification failed: ' + wheel.name)
            for member in archive.infolist():
                relative = PurePosixPath(member.filename)
                if (relative.is_absolute() or '..' in relative.parts or '\\' in member.filename
                    or not relative.parts or ((member.external_attr >> 16) & 0o170000) == 0o120000
                    or not (relative.parts[0] == name or
                            relative.parts[0].startswith(name + '-') and relative.parts[0].endswith('.dist-info'))):
                    raise ValueError('Unexpected original wheel member: ' + member.filename)
                target = output.joinpath(*relative.parts)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as stream:
                    stream.write(archive.read(member))
                count += 1
    record = dict(status='PINNED_ORIGINAL_MODEL_DEPENDENCY_FILES_STAGED', files=count,
        gpu_operations_executed=False, model_inference_executed=False,
        wheels={name + '==' + version: digest for name, (version, digest) in PINS.items()})
    (output / 'staged-dependencies.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
