"""Fetch public modular-checkpoint metadata only; never download weight shards."""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi,hf_hub_download


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    repo='MiniMaxAI/MiniMax-H3';info=HfApi().model_info(repo,files_metadata=True)
    selected=[];components={};records=[]
    prefixes=('transformer/','text_encoder/','vae/','audio_vae/')
    for file in info.siblings:
        name=file.rfilename
        if name in ('model_index.json','modular_model_index.json') or (name.startswith(prefixes) and
            (name.endswith('/config.json') or name.endswith('.safetensors.index.json'))):selected.append(name)
        if name.startswith(prefixes) and name.endswith('.safetensors'):
            component=name.split('/')[0]
            size=file.size
            if size is None:raise ValueError(f'Missing weight size for {name}')
            row=components.setdefault(component,dict(weight_file_bytes=0,shards=0))
            row['weight_file_bytes']+=size;row['shards']+=1
    for name in selected:
        file=Path(hf_hub_download(repo,name,revision=info.sha,local_dir=a.output))
        records.append(dict(file=name,sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    result=dict(repo=repo,revision=info.sha,metadata_only=True,weight_shards_downloaded=False,
                public_modular_weight_files=components,metadata_files=records,
                total_weight_file_bytes=sum(c['weight_file_bytes'] for c in components.values()),
                organizer_input_and_reference_confirmed=False)
    (a.output/'metadata-manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
