"""Offline model-path compatibility and immutable asset-view preparation."""
import argparse,ast,copy,hashlib,json,sys,tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from preflight import fl2va_model_directory
from prepare_model_view import prepare

def run(reference):
    tree=ast.parse(reference.read_text(encoding='utf-8'))
    selected=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and
              node.name in ('ModelPaths','_read_model_index','resolve_model_paths')]
    assert len(selected)==3
    namespace=dict(Path=Path,Any=Any,dataclass=dataclass,json=json,__name__=__name__)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(reference),'exec'),namespace)
    checks=[]
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);source=root/'original/diffusers';source.mkdir(parents=True)
        contents={'model_index.json':json.dumps({'_class_name':'MiniMaxH3ModularPipeline'}).encode(),
            'transformer/shard.safetensors':b'test','text_encoder/config.json':b'{}'}
        for name,data in contents.items():
            file=source/name;file.parent.mkdir(exist_ok=True);file.write_bytes(data)
        for path in (source,source.parent,source.parent/'FL2VA',source.parent/'Ref2VA'):
            actual=fl2va_model_directory(path)
            assert actual==namespace['resolve_model_paths'](str(path),'fl2va').modular_dir
        checks.append('all four path aliases match the unmodified reference resolver')
        (source/'diffusers').mkdir()
        assert fl2va_model_directory(source)==source
        checks.append('a directory named diffusers is not resolved twice')
        expected=dict(repo='MiniMaxAI/MiniMax-H3',revision='42ed227ee7df40d41602854ae760620d6eb651fe',
            files=[dict(file=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) for name,data in contents.items()])
        view=root/'view';result=prepare(source,view,expected)
        assert result['weight_files_copied'] is False and result['original_files_modified'] is False
        for name,data in contents.items():assert (source/name).read_bytes()==data and (view/name).read_bytes()==data
        assert (view/'transformer').is_symlink() and not (source/'asset-download-manifest.json').exists()
        assert namespace['resolve_model_paths'](str(view),'fl2va').modular_dir==view
        manifest=json.loads((view/'asset-download-manifest.json').read_text())
        assert manifest['complete'] and all(row['verified_sha256']==row['sha256'] for row in manifest['files'])
        checks.append('view links original bytes, records actual hashes and leaves the source unchanged')
        try:prepare(source,view,expected)
        except FileExistsError:checks.append('existing view cannot be overwritten')
        else:raise AssertionError('Existing view accepted')
        try:prepare(source,source/'nested-view',expected)
        except ValueError:checks.append('view cannot be created inside original assets')
        else:raise AssertionError('Original directory could be changed')
        (source/'transformer/shard.safetensors').write_bytes(b'evil')
        corrupt=root/'corrupt-view'
        try:prepare(source,corrupt,expected)
        except ValueError:
            assert not corrupt.exists();checks.append('same-size weight corruption rejected before creating a view')
        else:raise AssertionError('Corrupt weight accepted')
        bad=copy.deepcopy(expected);bad['files'][0]['file']='../outside'
        try:prepare(source,root/'unsafe-view',bad)
        except ValueError:checks.append('manifest path traversal rejected')
        else:raise AssertionError('Unsafe path accepted')
    return dict(status='PASS',checks=checks,metadata_and_small_byte_fixtures_only=True,full_fl2va_tested=False,
        reference_entry_sha256=hashlib.sha256(reference.read_bytes()).hexdigest())

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--reference-entry',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=run(args.reference_entry);args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result))
