"""Route the shared Qwen vision MHA through the optional Nine callback."""
import argparse,difflib,hashlib,json,shutil
from pathlib import Path

def patch(repo,package):
    path=repo/'csrc/models/qwen3_5/qwen3_5_vision.cpp'
    backup=path.with_name(path.name+'.h3-condition-reference')
    if not backup.exists():shutil.copy2(path,backup)
    original=backup.read_text()
    include='#include "qwen3_5_vision.hpp"'
    assert original.count(include)==1
    text=original.replace(include,include+'\n#include "../minimax_h3/h3_nine_bridge.hpp"',1)
    needle='''    auto out = infinicore::op::mha(q->unsqueeze(0), k->unsqueeze(0), v,
                                   std::nullopt, scale_, false)
                   ->view({seq_len, hidden_size_});'''
    assert original.count(needle)==1,'Reference Qwen vision attention changed'
    text=text.replace(needle,'''    auto query = q->unsqueeze(0);
    auto key = k->unsqueeze(0);
    infinicore::Tensor attended;
    if (!infinilm::h3_nine::try_compute("attention", query, key, v, attended))
        attended = infinicore::op::mha(query, key, v, std::nullopt, scale_, false);
    auto out = attended->view({seq_len, hidden_size_});''',1)
    if path.read_text() not in (original,text):raise RuntimeError('Unrecognized Qwen vision changes; merge manually')
    path.write_text(text)
    shutil.copy2(package/'native/h3_nine_bridge.hpp',repo/'csrc/models/minimax_h3/h3_nine_bridge.hpp')
    rows=[dict(file=path.relative_to(repo).as_posix(),reference_sha256=hashlib.sha256(original.encode()).hexdigest(),modified_sha256=hashlib.sha256(text.encode()).hexdigest())]
    (package/'native/conditioner-patch-manifest.json').write_text(json.dumps(rows,indent=2))
    (package/'native/InfiniLM-conditioner-integration.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile='a/'+rows[0]['file'],tofile='b/'+rows[0]['file'])))
    print('CONDITIONER_PATCH_APPLIED',repo)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--repo',type=Path,required=True);a=parser.parse_args()
    patch(a.repo.resolve(),Path(__file__).resolve().parents[1])
