"""Batch mode propagation and failure retention with deliberately missing assets."""
import argparse,json,os,subprocess,sys,tempfile
from pathlib import Path

def run():
    script=Path(__file__).resolve().parents[1]/'run_batch.py'
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);cases=root/'cases.jsonl'
        cases.write_text('\n'.join(json.dumps(dict(model_path=str(root/f'missing-{i}'),
            first_frame=str(root/'missing.png'),text=f'public negative case {i}')) for i in range(2)))
        results=root/'batch';env=dict(os.environ);env.pop('H3_WRAPPER_CHILD',None)
        result=subprocess.run([sys.executable,str(script),'--cases',str(cases),'--results',str(results),
            '--infinilm-root',str(root/'missing-reference'),'--experimental-streaming',
            '--experimental-matrix-attention','--preflight-only'],capture_output=True,text=True,env=env,timeout=60)
        assert result.returncode==1,(result.stdout,result.stderr)
        summary=json.loads((results/'batch.json').read_text())
        assert len(summary['cases'])==2 and all(row['status']=='FAILED' for row in summary['cases'])
        for i in range(2):
            folder=results/f'{i:04d}'
            audit=json.loads((folder/'video.mp4.run.json').read_text())
            assert audit['experimental_matrix_attention_requested'] and audit['experimental_streaming_requested']
            assert not (folder/'video.mp4').exists()
            assert all((folder/name).exists() for name in ('input.json','stdout.log','stderr.log'))
        blocked=root/'invalid-mode'
        result=subprocess.run([sys.executable,str(script),'--cases',str(cases),'--results',str(blocked),
            '--infinilm-root',str(root),'--experimental-matrix-attention'],capture_output=True,text=True,env=env,timeout=30)
        assert result.returncode==2 and not blocked.exists()
    return dict(status='PASS',host_negative_test_only=True,full_fl2va_tested=False,
        checks=['both optimization flags propagated','both failed cases retained','no video fabricated','unsupported mode rejected before results creation'])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=run();args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result))
