"""Verify supplemental development evidence without qualifying a benchmark run."""
import hashlib
import json
from pathlib import Path


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked_segmented_evidence(workspace):
    root = workspace/'outputs/completed-development-evidence-20261022'
    capture = json.loads((root/'capture-manifest.json').read_text(encoding='utf-8'))
    if capture.get('development_only') is not True or capture.get('performance_valid') is not False:
        raise ValueError('Capture must be declared development-only')
    files = {}
    for row in capture['files']:
        path = (root/row['file']).resolve()
        if not path.is_relative_to(root.resolve()) or path.is_symlink():
            raise ValueError('Unsafe evidence path')
        if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('Captured evidence changed: '+row['file'])
        files[row['file']] = path
    current = json.loads(files['sessions/current-stage.json'].read_text(encoding='utf-8'))
    session_name = Path(current['session_directory']).name
    session = 'sessions/'+session_name+'/'
    audit_path = files[session+'generated-development.mp4.run.json']
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    media_path = files[session+'media-spec.json']
    media = json.loads(media_path.read_text(encoding='utf-8'))
    supervisor = json.loads(files[session+'generated-development.session.json'].read_text(encoding='utf-8'))
    checkpoint_path = files['checkpoint-metadata/latest.json']
    checkpoint = json.loads(checkpoint_path.read_text(encoding='utf-8'))
    expected = dict(steps=50, num_frames=124, width=1344, height=768, seed=42, gpus='0')
    checks = [audit.get('status') == 'SEGMENTED_DEVELOPMENT_OUTPUT_UNVALIDATED',
              audit.get('segmented_self_test_complete') is True,
              audit.get('development_only') is True, audit.get('performance_valid') is False,
              audit.get('full_fl2va_tested') is False,
              audit.get('competition_passed') is False,
              audit.get('completed_iterations') == audit.get('total_iterations') == 49,
              checkpoint.get('completed_iterations') == checkpoint.get('total_iterations') == 49,
              audit.get('checkpoint_commit_sha256') == sha(checkpoint_path),
              supervisor.get('returncode') == 0,
              files[session+'runner.exit'].read_text().strip() == '0',
              media.get('status') == 'DEVELOPMENT_MEDIA_SPEC_PASS_QUALITY_UNVALIDATED',
              bool(media.get('checks')) and all(v is True for v in media['checks'].values()),
              media.get('generator_audit_sha256') == sha(audit_path),
              media.get('development_only') is True, media.get('performance_valid') is False,
              media.get('competition_passed') is False,
              media.get('organizer_similarity_checked') is False]
    for key, value in expected.items():
        checks.append(audit.get('requested_configuration', {}).get(key) == value)
        checks.append(checkpoint.get('identity', {}).get(key) == value)
    if not all(checks):
        raise ValueError('Incomplete or inconsistent segmented evidence')
    for name, digest in checkpoint['identity']['source'].items():
        captured = files['executed-source/submission/'+name]
        if sha(captured) != digest:
            raise ValueError('Executed source does not match checkpoint identity: '+name)
        # Packaging and documentation may evolve; model, runtime and entry code
        # must still match the exact source which produced this example.
        if name.startswith(('runtime/', 'native/', 'streaming/')) or name in (
                'run_fl2va.py', 'preflight.py', 'run_batch.py', 'inspect_output.py', 'validate_output.py'):
            if sha(workspace/'submission'/name) != digest:
                raise ValueError('Delivered inference source differs: '+name)
    for name, digest in audit.get('runtime_addendum', {}).items():
        if sha(files['development-tools/'+name]) != digest:
            raise ValueError('Development recovery source differs: '+name)
    example = workspace/'outputs/development-checkpoint-20261021/generated-development.mp4'
    if sha(example) != media['output_sha256']:
        raise ValueError('Development video hash mismatch')
    files['capture-manifest.json'] = root/'capture-manifest.json'
    entries = [('evidence/segmented-development/'+name, path) for name, path in sorted(files.items())]
    entries += [('examples/development-only/generated-development.mp4', example),
                ('examples/development-only/media-spec.json', media_path)]
    for name in ('generated-development-contact-sheet.jpg', 'local-video-inspection.json'):
        p = example.with_name(name)
        if p.exists():
            entries.append(('examples/development-only/'+name, p))
    state = dict(status='SEGMENTED_SELF_TEST_VIDEO_AVAILABLE', date='2026-10-22',
                 segmented_public_generation_completed=True,
                 segmented_media_spec_validated=True, segmented_generated_example_included=True,
                 segmented_output_sha256=media['output_sha256'],
                 segmented_finished_at=supervisor.get('finished_at'),
                 inference_source_matches_development_run=True,
                 latest_segmented_evidence='evidence/segmented-development/',
                 full_fl2va_tested=False, ready_for_preliminary_submission=False,
                 organizer_similarity_validated=False, continuous_performance_validated=False,
                 limitations=['Only segmented development generation has completed',
                              'Organizer reference similarity has not been evaluated',
                              'Uninterrupted full generation and steady-state timing remain unverified',
                              'Multi-GPU execution and a second clean offline rebuild remain unverified'])
    return entries, state
