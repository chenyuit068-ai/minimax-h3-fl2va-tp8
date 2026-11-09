"""A native-style worker termination must not leave a RUNNING audit."""
import json,os,shutil,subprocess,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from h3_process_supervisor import supervise

if __name__=='__main__':
    with tempfile.TemporaryDirectory() as temp:
        audit=Path(temp)/'run.json'
        child='''import json,os,signal,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps(dict(status='RUNNING',invocation_id=os.environ['H3_INVOCATION_ID'])))
os.kill(os.getpid(),signal.SIGTERM)
'''
        code=supervise([sys.executable,'-c',child,str(audit)],audit)
        data=json.loads(audit.read_text())
        assert code and data['status']=='FAILED_PROCESS_TERMINATION'
        assert data['full_fl2va_tested'] is False and data['competition_passed'] is False
        previous=audit.read_bytes()
        assert supervise([sys.executable,'-c','raise SystemExit(2)'],audit)==2
        assert audit.read_bytes()==previous
        early=child.replace('os.kill(os.getpid(),signal.SIGTERM)','raise SystemExit(0)')
        assert supervise([sys.executable,'-c',early,str(audit)],audit)==1
        assert json.loads(audit.read_text())['status']=='FAILED_PROCESS_TERMINATION'
        # Execute the actual wrapper with a host-only fake preflight and entry.
        # This checks the audit is persisted before calling the model entry,
        # rather than merely testing a worker that writes RUNNING itself.
        package=Path(__file__).resolve().parents[1]
        facade=root=Path(temp)/'facade';(facade/'runtime').mkdir(parents=True)
        shutil.copy2(package/'run_fl2va.py',facade/'run_fl2va.py')
        shutil.copy2(package/'runtime/h3_process_supervisor.py',facade/'runtime/h3_process_supervisor.py')
        (facade/'preflight.py').write_text("def check(*args,**kwargs):return dict(status='PREFLIGHT_ONLY_PASS',errors=[],devices=[])\n")
        repo=Path(temp)/'fake-model-entry';(repo/'examples').mkdir(parents=True)
        (repo/'examples/minimax_h3.py').write_text('''import json,os,signal,sys
from pathlib import Path
path=Path(next(x.split('=',1)[1] for x in sys.argv if x.startswith('--output=')))
assert json.loads(path.with_suffix(path.suffix+'.run.json').read_text())['status']=='RUNNING'
os.kill(os.getpid(),signal.SIGTERM)
''')
        media=Path(temp)/'native-signal.mp4'
        env=dict(os.environ);env.pop('H3_WRAPPER_CHILD',None)
        probe=subprocess.run([sys.executable,str(facade/'run_fl2va.py'),'fl2va',
            '--model-path',str(root),'--first-frame',str(root/'unused.png'),'--text','host-only test',
            '--output',str(media),'--infinilm-root',str(repo)],env=env,capture_output=True,text=True,timeout=30)
        assert probe.returncode!=0,(probe.stdout,probe.stderr)
        data=json.loads(media.with_suffix('.mp4.run.json').read_text())
        assert data['status']=='FAILED_PROCESS_TERMINATION',(data,probe.stderr)
    print(json.dumps(dict(status='PASS',checks=['worker signal recorded as failure','earlier invocation audit unchanged','zero exit without completion rejected','actual wrapper persists running audit before model entry'],full_fl2va_tested=False)))
