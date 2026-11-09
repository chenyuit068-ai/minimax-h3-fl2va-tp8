"""Build one auditable source ZIP; optionally include the existing offline bundle.

This deliberately never declares preliminary completion without full evidence.
No model weights, native binaries or credentials are collected. A completed
public generated example is included only after its audit/spec/hash checks pass.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def build(workspace, output, include_offline=False, include_development=False, baseline_override=None):
    package=workspace/'submission'
    baseline=Path(baseline_override) if baseline_override is not None else workspace/'outputs/MiniMax-H3-完整离线复现包.zip'
    if output.exists():raise FileExistsError('Use a new candidate archive name')
    checksums=[]
    development_entries=[]
    development_state={}
    if include_development:
        from segmented_evidence import checked_segmented_evidence
        development_entries,development_state=checked_segmented_evidence(workspace)
    def add(z,name,data):
        z.writestr(name,data)
        checksums.append(dict(file=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(package.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:
                if p.suffix not in ('.py','.sh','.md','.hpp','.h','.cpp','.cc','.patch','.json','.txt','.csv'):
                    raise ValueError(f'Unexpected source artifact: {p}')
                add(z,p.relative_to(workspace).as_posix(),p.read_bytes())
        add(z,'README.md',package.joinpath('README.md').read_bytes())
        first=workspace/'work/public-inputs/official-demo-extracted-first-frame.png'
        data=first.read_bytes()
        first_sha='2c302d70b5b015db1640cf0ebff8039409fda882b528ba31b9b6e8f4fb698086'
        if hashlib.sha256(data).hexdigest()!=first_sha:raise ValueError('Public first-frame hash mismatch')
        add(z,'examples/public-self-test/first-frame.png',data)
        provenance=dict(official_competition_case=False,official_reference_video=False,
            source='https://raw.githubusercontent.com/MiniMax-AI/MiniMax-H3/d21241f0a4b3acbb34c97dae47fa417b7065e438/assets/i2va.mp4',
            source_sha256='5a5d6e0fcdda7e6a5d98cb92d9e5cb4b6b0b44d6e1dfd5a370d16dfaa79b9ae6',
            extracted_frame=0,first_frame_sha256=first_sha,width=1344,height=768,
            purpose='Public functional self-test input only; never reused as generated output')
        add(z,'examples/public-self-test/provenance.json',json.dumps(provenance,indent=2).encode())
        preprocessing=json.loads((package/'auxiliary-dependencies.json').read_text())['torchvision']
        wheel=workspace/'work/scnet-stage/wheels'/preprocessing['filename']
        data=wheel.read_bytes()
        if hashlib.sha256(data).hexdigest()!=preprocessing['sha256']:raise ValueError('Preprocessing wheel hash mismatch')
        add(z,'auxiliary-wheelhouse/'+wheel.name,data)
        with zipfile.ZipFile(baseline) as old:
            # Preserve original source snapshots, operator evidence and scripts.
            # Old report and email material remain historical under baseline/.
            for entry in old.infolist():
                if entry.is_dir() or entry.filename.startswith('tools/'):continue
                if entry.filename.startswith('wheelhouse/') and 'ninetoothed-0.26.0-' not in entry.filename:continue
                if entry.filename=='application-email.txt':continue
                add(z,'baseline/'+entry.filename,old.read(entry))
            add(z,'source-manifest.json',old.read('source-manifest.json'))
        # The lightweight baseline is for source review. Its old full-bundle
        # checksum list cannot be used to rebuild until the offline ZIP is used.
        for pattern in ('outputs/*local.json','outputs/submission-host-tools-tests.json','outputs/batch-negative-tests.json',
                        'outputs/native-evidence-current/**/*','outputs/attention-evidence-20261020/**/*',
                        'outputs/component-evidence-20261020-*/**/*',
                        'outputs/component-evidence-20261020-*-manifest.json',
                        'outputs/next-attention-evidence-20261020/**/*',
                        'outputs/resumed-target-evidence-*/**/*',
                        'outputs/resumed-source-match-*.json',
                        'outputs/mm-tile-screening-evidence-20261020/**/*',
                        'outputs/*-host-tests-20261020.json',
                        'outputs/weight-store-tests-20261020.json','outputs/official-weight-store-read-20261020.json',
                        'outputs/*local-20261020.json',
                        'outputs/process-supervisor-tests-20261020*.json',
                        'outputs/completed-submission-*-20261022.json',
                        'outputs/optimization-evidence-20261023/**/*',
                        'outputs/selected-runtime-import-20261023.json',
                        'outputs/pv-fused-runtime-import-20261023.json',
                        'outputs/optimization-release-20261023.json',
                        'outputs/release-source-verification-20261023.json',
                        'outputs/local-model-cache-host-tests-20261023.json',
                        'outputs/source-audit-host-tests-20261023.json',
                        'outputs/clean-rebuild-materialized-bundle-20261023.json',
                        'outputs/clean-rebuild-script-fix-20261023.json',
                        'outputs/generated-video-first-frame-diagnostic-20261023.json',
                        'outputs/*恢复检查*.md',
                        'outputs/public-model-metadata/**/*.json'):
            for p in sorted(workspace.glob(pattern)):
                if p.is_file() and p.suffix in ('.json','.log','.md','.exit','.txt','.patch','.hpp','.h','.cpp','.cc','.py','.sh','.source','.llir','.amdgcn'):
                    name='evidence/current/'+p.relative_to(workspace/'outputs').as_posix()
                    if name not in {x['file'] for x in checksums}:add(z,name,p.read_bytes())
        proof=workspace/'outputs/SCNET-加速卡资源不足-20261020.png'
        if proof.is_file():add(z,'evidence/platform/insufficient-accelerator-20261020.png',proof.read_bytes())
        retry_proof=workspace/'outputs/SCNET-加速卡资源不足-20261020-复查.png'
        if retry_proof.is_file():add(z,'evidence/platform/insufficient-accelerator-recheck-20261020.png',retry_proof.read_bytes())
        for name in ('h3-adaptive-dispatch-local.log','h3-fast-dispatch-local.log','h3-local-attention-integration.log','h3-native-benchmark-local-final.log',
                     'h3-local-final-tests-v2.log','h3-attention-local.log'):
            p=workspace/'work/logs'/name
            if p.is_file():add(z,'evidence/local-logs/'+name,p.read_bytes())
        for name in ('h3-native-attention.zip','h3-vector-attention.zip','h3-vector-one-key.zip','h3-native-fast.zip'):
            path=workspace/'work/scnet-stage'/name
            if path.is_file():add(z,'evidence/experiment-source/'+name,path.read_bytes())
        state=dict(schema_version=1,status='DEVELOPMENT_CANDIDATE',team='从容应对',captain='徐晨宇',
                   date='2026-10-20',full_fl2va_tested=False,ready_for_preliminary_submission=False,
                   native_layer_offload_implemented=True,official_weights_included=False,
                   native_layer_storage_experimental=True,full_phase_generation_validated=False,
                   hip_attention_validated_on_component_cases=True,
                   hip_full_video_attention_validated=False,
                   environment_status='UNKNOWN_CHECK_LATEST_EVIDENCE',
                   offline_dependencies_included=include_offline,
                   limitations=['No full official FL2VA video/reference validation',
                                'Resident loading exceeds the assigned single GPU; experimental offload is required',
                                'Full video attention performance and multi-GPU behavior unverified'])
        # Derive the captured state from actual target audit files. A running
        # process or passed component test must never become a completed video.
        # A changed numerical backend cannot inherit completion or RUNNING
        # state from older source captures. Its own full run is still required.
        backend_path=package/'attention-backend-manifest.json'
        audits=[] if backend_path.is_file() else list(workspace.glob('outputs/resumed-target-evidence-*/full-generation/generated-v3.mp4.run.json'))
        if backend_path.is_file():
            backend=json.loads(backend_path.read_text(encoding='utf-8'))
            for name,digest in backend['runtime_sources'].items():
                if Path(name).name!=name:raise ValueError('Invalid declared runtime path')
                if hashlib.sha256((package/'runtime'/name).read_bytes()).hexdigest()!=digest:
                    raise ValueError('Selected runtime source differs from its target evidence: '+name)
            state.update(status='OPTIMIZED_COMPONENT_CANDIDATE_FULL_VIDEO_PENDING',date='2026-10-23',
                selected_attention_backend=backend['selected_backend'],
                numerical_source_changed_after_v4=True,full_public_generation_completed=False,
                public_generated_example_included=False,organizer_similarity_validated=False,
                full_phase_generation_validated=False,continuous_performance_validated=False)
            if backend['selected_backend']=='pv_fused':
                imported=json.loads((workspace/'outputs/pv-fused-runtime-import-20261023.json').read_text(encoding='utf-8'))
                target=workspace/'outputs/optimization-evidence-20261023/pv-fused-integration.json'
                if hashlib.sha256(target.read_bytes()).hexdigest()!=imported['target_integration_evidence_sha256']:
                    raise ValueError('Integration evidence does not match the imported runtime')
                proof=json.loads(target.read_text(encoding='utf-8'))
                if proof['status']!='PASS' or not all(c['status']=='PASS' for c in proof['checks'].values()):
                    raise ValueError('Selected backend integration is not qualified')
                identity=imported['source_identity']
                if identity!=proof['checks']['imports']['source_identity']:
                    raise ValueError('Imported source identity differs from target integration')
                for name,digest in identity['inference_sources'].items():
                    if hashlib.sha256((package/name).read_bytes()).hexdigest()!=digest:
                        raise ValueError('Inference source changed after qualification: '+name)
                if hashlib.sha256(backend_path.read_bytes()).hexdigest()!=identity['attention_backend_manifest_sha256']:
                    raise ValueError('Backend manifest differs from target qualification')
                state.update(selected_backend_component_integration_validated=True,
                    independent_native_build_validated=True,pipeline_metadata_only_validated=True,
                    qualified_inference_source_files=len(identity['inference_sources']),
                    integration_evidence_sha256=imported['target_integration_evidence_sha256'],
                    environment_status='SINGLE_DCU_VERIFIED_ON_2026_10_23_NOT_A_LIVE_ALLOCATION_CHECK',
                    limitations=['No full continuous video from the current numerical source',
                                 'No official standard video or similarity threshold provided',
                                 'Main attention estimate alone exceeds the approximately 8-hour resource window',
                                 'Steady multi-case resource usage and multi-GPU behavior unverified'])
        if audits:
            def capture_time(path):
                manifests=list(path.parents[1].glob('*.tar.manifest.json'))
                return datetime.datetime.fromisoformat(json.loads(manifests[0].read_text(encoding='utf-8'))['captured_at']).timestamp() if len(manifests)==1 else 0
            latest=max(audits,key=lambda path:(capture_time(path),str(path)))
            audit=json.loads(latest.read_text(encoding='utf-8'))
            completed=(audit.get('status')=='GENERATION_COMPLETED_UNVALIDATED' and
                       audit.get('full_fl2va_tested') is True)
            state.update(environment_status='SINGLE_DCU_RESTORED_AS_OF_CAPTURE',
                status='FULL_PUBLIC_SELF_TEST_RUNNING' if audit.get('status')=='RUNNING' else
                       'PUBLIC_GENERATION_COMPLETED_QUALITY_UNVALIDATED' if completed else 'DEVELOPMENT_CANDIDATE',
                latest_public_run_status=audit.get('status'),
                latest_public_run_evidence=latest.relative_to(workspace/'outputs').as_posix(),
                full_fl2va_tested=completed,full_public_generation_completed=completed,
                organizer_similarity_validated=False)
            folder=latest.parents[1]
            media=folder/'integration/generated-v3-media-spec.json'
            media_ok=False
            if completed and media.is_file():
                inspection=json.loads(media.read_text(encoding='utf-8'))
                media_ok=(inspection.get('status')=='MEDIA_SPEC_PASS_QUALITY_UNVALIDATED' and
                          bool(inspection.get('checks')) and all(inspection['checks'].values()) and
                          inspection.get('generator_audit_sha256')==hashlib.sha256(latest.read_bytes()).hexdigest())
            state['public_media_spec_validated']=media_ok
            state['full_phase_generation_validated']=media_ok
            state['public_generated_example_included']=False
            if media_ok:
                example=workspace/'outputs/public-fl2va-v3/generated-v3.mp4'
                if not example.is_file():raise FileNotFoundError('Download the completed public output before packaging')
                if hashlib.file_digest(example.open('rb'),'sha256').hexdigest()!=inspection['output_sha256']:
                    raise ValueError('Public generated example differs from the decoded target output')
                add(z,'examples/public-self-test/generated-v3.mp4',example.read_bytes())
                add(z,'examples/public-self-test/media-spec.json',media.read_bytes())
                state['public_generated_example_included']=True
            qualifications=[folder/'qualification'/name for name in
                ('exp2-range.json','short-96.json','power2-long-56-heads.json','compact-long-56-heads.json')]
            state['hip_matrix_attention_target_component_validated']=all(
                p.is_file() and json.loads(p.read_text(encoding='utf-8')).get('status')=='PASS' for p in qualifications)
            manifests=list(folder.glob('*.tar.manifest.json'))
            if len(manifests)==1:
                state['latest_capture_at']=json.loads(manifests[0].read_text(encoding='utf-8'))['captured_at']
                state['source_snapshot_date']=state['date']
                state['date']=datetime.datetime.fromisoformat(state['latest_capture_at']).astimezone(
                    datetime.timezone(datetime.timedelta(hours=8))).date().isoformat()
        # Supplemental segmented output never satisfies the ordinary continuous
        # generation/spec gate above, or the organizer's quality requirements.
        if development_state:
            state.update(development_state)
            for name,path in development_entries:add(z,name,path.read_bytes())
        add(z,'completion-status.json',json.dumps(state,ensure_ascii=False,indent=2).encode())
        if include_offline:
            # Already-compressed wheels are stored without recompression.
            info=zipfile.ZipInfo('baseline-offline.zip');info.compress_type=zipfile.ZIP_STORED
            with baseline.open('rb') as source,z.open(info,'w',force_zip64=True) as dest:
                shutil.copyfileobj(source,dest,1024*1024)
            checksums.append(dict(file=info.filename,bytes=baseline.stat().st_size,
                sha256=hashlib.file_digest(baseline.open('rb'),'sha256').hexdigest()))
        z.writestr('SHA256SUMS.json',json.dumps(checksums,ensure_ascii=False,indent=2).encode())
    # Decompress, CRC-check and hash every archived file before delivery.
    with zipfile.ZipFile(output) as z:
        assert z.testzip() is None
        for row in checksums:
            with z.open(row['file']) as f:
                assert hashlib.file_digest(f,'sha256').hexdigest()==row['sha256'],row['file']
    digest=hashlib.file_digest(output.open('rb'),'sha256').hexdigest()
    output.with_suffix(output.suffix+'.sha256').write_text(digest+'  '+output.name+'\n',encoding='utf-8')
    print(json.dumps(dict(file=str(output),bytes=output.stat().st_size,sha256=digest,
                         verified_entries=len(checksums),full_fl2va_tested=state['full_fl2va_tested'],
                         ready_for_preliminary_submission=state['ready_for_preliminary_submission']),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--include-offline',action='store_true')
    p.add_argument('--baseline-offline',type=Path,
                   help='Explicit verified baseline bundle, for example the corrected offline build bundle')
    p.add_argument('--include-development-evidence',action='store_true',
                   help='Include verified segmented self-test evidence without qualifying it for scoring')
    a=p.parse_args()
    build(Path(__file__).resolve().parents[2],a.output.resolve(),a.include_offline,a.include_development_evidence,a.baseline_offline)
