"""Wire the offline pipeline without loading or running numerical model phases."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time


def run(args):
    if args.output.exists():
        raise FileExistsError('Choose a fresh metadata check report')
    package = Path(__file__).resolve().parents[1]
    entry = args.infinilm_root/'examples/minimax_h3.py'
    sys.path.insert(0, str(package/'runtime'))
    from h3_source_audit import capture
    identity = capture(package, entry)
    from h3_phase_pipeline import make_pipeline
    import torch
    import infinicore as core
    core.set_device(core.device('cuda', args.device))
    torch.cuda.set_device(args.device)
    name = 'h3_metadata_check_reference'
    spec = importlib.util.spec_from_file_location(name, entry)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    paths = module.resolve_model_paths(str(args.model_path), 'fl2va')
    started = time.perf_counter()
    pipe, text, transformer = make_pipeline(module, paths, [args.device], {}, metadata_only=True)
    phases = pipe._h3_phase_residency
    try:
        assert not phases.transitions and phases.active_model is None
        result = dict(status='PASS', pipeline_class=type(pipe).__name__,
            model_path=str(paths.modular_dir), device=torch.cuda.get_device_name(args.device),
            phase_factories=sorted(phases.factories), model_phases_loaded=False,
            full_fl2va_tested=False, numerical_model_computation_executed=False,
            source_identity=identity, setup_wall_seconds=time.perf_counter()-started)
    finally:
        phases.release()
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--infinilm-root', type=Path, required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    print(json.dumps(run(parser.parse_args())), flush=True)
