"""Verify existing original assets offline and link a separate model view.

Never modifies or copies weights. Useful when the organizer's model directory
is read-only or does not contain this project's download verification manifest.
"""
import argparse,datetime,hashlib,json,os
from pathlib import Path,PurePosixPath
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from preflight import fl2va_model_directory

def prepare(source,output,expected):
    source=fl2va_model_directory(source);output=Path(output).expanduser().resolve()
    if output.exists():raise FileExistsError('Use a new model-view directory')
    if source==output or source in output.parents:raise ValueError('The model view must be outside the original model directory')
    if output.name in ('FL2VA','Ref2VA'):raise ValueError('Choose a view name without the reference entry\'s partition aliases')
    if not output.parent.is_dir():raise FileNotFoundError('Create the requested parent directory first')
    verified=[];tops=set()
    for row in expected['files']:
        relative=PurePosixPath(row['file'])
        if relative.is_absolute() or '..' in relative.parts or not relative.parts:
            raise ValueError('Unsafe manifest path')
        path=source/relative
        if not path.is_file() or path.stat().st_size!=row['bytes']:
            raise ValueError('Original asset missing or size differs: '+row['file'])
        with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=row['sha256']:raise ValueError('Original asset SHA256 differs: '+row['file'])
        verified.append(dict(row,verified_sha256=actual));tops.add(relative.parts[0])
        print('ASSET_VERIFIED',row['file'],flush=True)
    # All file hashes must pass before creating any part of the new view.
    output.mkdir()
    for top in sorted(tops):
        target=source/top
        os.symlink(target,output/top,target_is_directory=target.is_dir())
    manifest=dict(repo=expected['repo'],revision=expected['revision'],files=verified,complete=True,
        verification='Full original file SHA256; no weight conversion',source_directory=str(source),
        verified_at=datetime.datetime.now().astimezone().isoformat())
    (output/'asset-download-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return dict(status='ORIGINAL_ASSETS_VERIFIED_VIEW_CREATED',model_path=str(output),files=len(verified),
        original_files_modified=False,weight_files_copied=False,full_fl2va_tested=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    reference=Path(__file__).resolve().parents[1]/'assets/original-model-files.json'
    expected=json.loads(reference.read_text(encoding='utf-8'))
    assert expected['repo']=='MiniMaxAI/MiniMax-H3' and expected['revision']=='42ed227ee7df40d41602854ae760620d6eb651fe'
    print(json.dumps(prepare(args.model_path,args.output,expected)),flush=True)
