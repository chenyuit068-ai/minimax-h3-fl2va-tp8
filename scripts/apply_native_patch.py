"""Install a minimal, idempotent patch without modifying the saved baseline."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil


def patch(repo, package):
    cpp = repo/'csrc/models/minimax_h3/minimax_h3_transformer.cpp'
    bindings = repo/'csrc/pybind11/bindings.cc'
    records = []
    for path in (cpp, bindings):
        original = path.with_suffix(path.suffix+'.h3-reference')
        if not original.exists(): shutil.copy2(path, original)
        old = original.read_text()
        if path == cpp:
            text = old.replace('#include "minimax_h3_transformer.hpp"', '#include "minimax_h3_transformer.hpp"\n#include "h3_nine_bridge.hpp"', 1)
            needle = '    auto scaled = infinicore::op::mul(normalized, scale);'
            assert old.count(needle) == 1, 'Reference AdaLN implementation changed'
            text = text.replace(needle, '    infinicore::Tensor fused;\n    if (infinilm::h3_nine::try_compute("adaln", normalized, scale, shift, fused)) return fused;\n'+needle, 1)
            helper = '''infinicore::Tensor apply_gated_residual(const infinicore::Tensor &x,
                                              const infinicore::Tensor &gate,
                                              const infinicore::Tensor &branch) {
    infinicore::Tensor fused;
    if (infinilm::h3_nine::try_compute("residual", x, gate, branch, fused)) return fused;
    return infinicore::op::add(x, infinicore::op::mul(gate, branch));
}

'''
            text = text.replace('void require_tensor(', helper+'void require_tensor(', 1)
            old_attn = 'auto output = infinicore::op::add(\n        hidden_states, infinicore::op::mul(gate_msa, attention_output));'
            old_mlp = 'output = infinicore::op::add(output, infinicore::op::mul(gate_mlp, mlp_output));'
            assert old_attn in old and old_mlp in old, 'Reference residual implementation changed'
            text = text.replace(old_attn, 'auto output = apply_gated_residual(hidden_states, gate_msa, attention_output);', 1)
            text = text.replace(old_mlp, 'output = apply_gated_residual(output, gate_mlp, mlp_output);', 1)
            attention = '    auto output = infinicore::op::mha(query, key, value, std::nullopt, scale_, false);'
            assert old.count(attention) == 1, 'Reference H3 attention implementation changed'
            text = text.replace(attention, '''    infinicore::Tensor output;
    if (!infinilm::h3_nine::try_compute("attention", query, key, value, output))
        output = infinicore::op::mha(query, key, value, std::nullopt, scale_, false);''', 1)
        else:
            text = old.replace('#include "engine/engine.hpp"', '#include "engine/engine.hpp"\n#include "../models/minimax_h3/h3_nine_bindings.hpp"', 1)
            assert '    infinilm::engine::bind_infer_engine(m);' in old
            text = text.replace('    infinilm::engine::bind_infer_engine(m);', '    infinilm::engine::bind_infer_engine(m);\n    infinilm::h3_nine::bind(m);', 1)
        # Do not overwrite unrelated user edits or an unrecognized patch.
        previous_hashes = set()
        reference_hash = hashlib.sha256(old.encode()).hexdigest()
        for previous in (package/'native/patch-manifest.json',package/'native/patch-manifest-history.json'):
            if previous.exists():
                previous_hashes.update(row['modified_sha256'] for row in json.loads(previous.read_text())
                    if row['file'] == str(path.relative_to(repo)) and row['reference_sha256'] == reference_hash)
        current = path.read_text()
        known_previous = hashlib.sha256(current.encode()).hexdigest() in previous_hashes
        assert current in (old, text) or known_previous, f'Unrelated edits in {path}; merge manually'
        path.write_text(text)
        records.append(dict(file=str(path.relative_to(repo)), reference_sha256=hashlib.sha256(old.encode()).hexdigest(),
                            modified_sha256=hashlib.sha256(text.encode()).hexdigest()))
    for name in ('h3_nine_bridge.hpp','h3_nine_bindings.hpp','h3_dlpack.h'):
        shutil.copy2(package/'native'/name, cpp.parent/name)
    diffs = []
    for row in records:
        path=repo/row['file']; original=path.with_suffix(path.suffix+'.h3-reference')
        diffs.extend(difflib.unified_diff(original.read_text().splitlines(True), path.read_text().splitlines(True),
                                        fromfile='a/'+row['file'],tofile='b/'+row['file']))
    (package/'native/InfiniLM-native-integration.patch').write_text(''.join(diffs))
    (package/'native/patch-manifest.json').write_text(json.dumps(records,indent=2))
    print('NATIVE_PATCH_APPLIED', repo)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--repo',type=Path,required=True)
    args=parser.parse_args(); patch(args.repo.resolve(),Path(__file__).resolve().parents[1])
