"""Decode a completed generated video and check specified media properties.

This does not replace comparison with the organizer's standard output.
"""
import argparse,hashlib,json
from fractions import Fraction
from pathlib import Path
import av
import numpy as np

def inspect(args):
    if args.report.exists():raise FileExistsError('Use a fresh inspection report')
    audit=json.loads(args.audit.read_text())
    checks=dict(generation_completed=audit.get('status')=='GENERATION_COMPLETED_UNVALIDATED'
                and audit.get('full_fl2va_tested') is True)
    frames=0;timestamps=[];dimensions=True
    with av.open(str(args.output)) as container:
        video=container.streams.video[0]
        metadata=dict(width=video.width,height=video.height,fps=str(video.average_rate or video.base_rate),
            video_codec=video.codec_context.name,pixel_format=video.codec_context.format.name,
            container=container.format.name)
        for frame in container.decode(video=0):
            frames+=1;dimensions=dimensions and frame.width==args.width and frame.height==args.height
            if frame.pts is not None:timestamps.append(float(frame.pts*frame.time_base))
    checks.update(width=metadata['width']==args.width,height=metadata['height']==args.height,
        all_decoded_dimensions=dimensions,decoded_frames=frames==args.frames,
        fps=Fraction(metadata['fps'])==Fraction(args.fps),video_codec=metadata['video_codec']==args.video_codec,
        pixel_format=metadata['pixel_format']==args.pixel_format,
        video_timestamps=len(timestamps)==frames and all(a<b for a,b in zip(timestamps,timestamps[1:])))
    audio_samples=0;finite=True;audio_metadata=None
    with av.open(str(args.output)) as container:
        if container.streams.audio:
            audio=container.streams.audio[0]
            audio_metadata=dict(codec=audio.codec_context.name,sample_rate=audio.codec_context.sample_rate,
                channels=len(audio.codec_context.layout.channels))
            for frame in container.decode(audio=0):
                audio_samples+=frame.samples
                finite=finite and bool(np.isfinite(frame.to_ndarray()).all())
    checks.update(audio_present=audio_metadata is not None,
        audio_sample_rate=audio_metadata is not None and audio_metadata['sample_rate']==args.audio_sample_rate,
        audio_channels=audio_metadata is not None and audio_metadata['channels']==args.audio_channels,
        audio_codec=audio_metadata is not None and audio_metadata['codec']==args.audio_codec,
        decoded_audio_nonempty=audio_samples>0,audio_finite=finite and audio_samples>0)
    result=dict(status='MEDIA_SPEC_PASS_QUALITY_UNVALIDATED' if all(checks.values()) else 'FAIL',
        checks=checks,decoded_frames=frames,decoded_audio_samples=audio_samples,video=metadata,audio=audio_metadata,
        video_first_timestamp=timestamps[0] if timestamps else None,
        video_last_timestamp=timestamps[-1] if timestamps else None,
        output_sha256=hashlib.file_digest(args.output.open('rb'),'sha256').hexdigest(),
        generator_audit_sha256=hashlib.sha256(args.audit.read_bytes()).hexdigest(),
        organizer_similarity_checked=False,competition_passed=False,
        note='Functional public self-test only; no organizer standard output or quality threshold was available.')
    args.report.write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    for name in ('width','height','frames'):parser.add_argument('--'+name,type=int,required=True)
    parser.add_argument('--fps',default='24');parser.add_argument('--video-codec',default='h264')
    parser.add_argument('--pixel-format',default='yuv420p');parser.add_argument('--audio-codec',default='aac')
    parser.add_argument('--audio-sample-rate',type=int,default=32000);parser.add_argument('--audio-channels',type=int,default=2)
    result=inspect(parser.parse_args());print(json.dumps(result),flush=True)
    raise SystemExit(0 if all(result['checks'].values()) else 2)
