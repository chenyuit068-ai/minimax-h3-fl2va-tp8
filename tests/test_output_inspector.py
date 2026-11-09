"""Synthetic media acceptance and rejection cases; never a generated-model result."""
import argparse,json,sys,tempfile
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import av
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from inspect_output import inspect

def run():
    cases=[]
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);video=root/'synthetic.mp4'
        with av.open(str(video),'w') as container:
            stream=container.add_stream('libx264',rate=24)
            stream.width=64;stream.height=64;stream.pix_fmt='yuv420p'
            audio=container.add_stream('aac',rate=32000);audio.layout='stereo'
            for i in range(8):
                pixels=np.zeros((64,64,3),dtype=np.uint8);pixels[:,:,0]=i*20
                frame=av.VideoFrame.from_ndarray(pixels,format='rgb24');frame.pts=i;frame.time_base=Fraction(1,24)
                for packet in stream.encode(frame):container.mux(packet)
            for packet in stream.encode():container.mux(packet)
            frame=av.AudioFrame.from_ndarray(np.zeros((2,10667),dtype=np.float32),format='fltp',layout='stereo')
            frame.sample_rate=32000;frame.pts=0;frame.time_base=Fraction(1,32000)
            for packet in audio.encode(frame):container.mux(packet)
            for packet in audio.encode():container.mux(packet)
        audit=root/'audit.json'
        audit.write_text(json.dumps(dict(status='GENERATION_COMPLETED_UNVALIDATED',full_fl2va_tested=True,synthetic_test_only=True)))
        defaults=dict(output=video,audit=audit,width=64,height=64,frames=8,fps='24',video_codec='h264',
            pixel_format='yuv420p',audio_codec='aac',audio_sample_rate=32000,audio_channels=2)
        for label,changes,field in [('valid',{},None),('wrong-frame-count',dict(frames=9),'decoded_frames'),
                                    ('wrong-dimensions',dict(width=32),'width'),('wrong-fps',dict(fps='30'),'fps')]:
            args=SimpleNamespace(**(defaults|changes),report=root/(label+'.json'))
            result=inspect(args)
            assert result['competition_passed'] is False and result['organizer_similarity_checked'] is False
            if field:assert result['status']=='FAIL' and result['checks'][field] is False
            else:assert result['status']=='MEDIA_SPEC_PASS_QUALITY_UNVALIDATED'
            cases.append(label)
        audit.write_text(json.dumps(dict(status='FAILED',full_fl2va_tested=False)))
        result=inspect(SimpleNamespace(**defaults,report=root/'failed-worker.json'))
        assert result['status']=='FAIL' and not result['checks']['generation_completed']
        cases.append('failed-generator-rejected')
    return dict(status='PASS',synthetic_media_only=True,full_fl2va_tested=False,checks=cases)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=run();args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result))
