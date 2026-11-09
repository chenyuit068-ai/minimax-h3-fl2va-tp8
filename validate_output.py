"""Independent stricter output checks; never modify the organizer comparator."""
import argparse
import json
from pathlib import Path
import runpy
import av


def metadata(path):
    with av.open(str(path)) as c:
        video=c.streams.video[0]
        audio=c.streams.audio[0] if c.streams.audio else None
        return dict(width=video.width,height=video.height,video_codec=video.codec_context.name,
                    fps=str(video.average_rate or video.base_rate),format=c.format.name,
                    audio=None if audio is None else dict(codec=audio.codec_context.name,
                        sample_rate=audio.codec_context.sample_rate,channels=len(audio.codec_context.layout.channels)))


def validate(reference,output,comparator,min_ssim):
    if not 0<=min_ssim<=100:raise ValueError('SSIM threshold must be in [0,100]')
    left,right=metadata(reference),metadata(output)
    metrics=runpy.run_path(str(comparator))['compare_video_streams'](Path(reference),Path(output))
    checks={name:left[name]==right[name] for name in ('width','height','video_codec','fps','format','audio')}
    checks['decoded_frames']=metrics['decoded_length_match'] and metrics['left_frames']>0
    checks['ssim']=metrics['ssim_match_percent'] is not None and metrics['ssim_match_percent']>=min_ssim
    return dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,reference=left,output=right,
                video_metrics=metrics,min_ssim_supplied_by_caller=min_ssim,
                organizer_threshold_confirmed=False,generation_provenance_checked=False,
                note='Checks media properties and reference similarity only; does not by itself prove rule-compliant generation or official acceptance')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--comparator',type=Path,required=True)
    p.add_argument('--min-ssim',type=float,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    r=validate(a.reference,a.output,a.comparator,a.min_ssim);a.report.write_text(json.dumps(r,indent=2));print(json.dumps(r))
    raise SystemExit(0 if r['status']=='PASS' else 2)
