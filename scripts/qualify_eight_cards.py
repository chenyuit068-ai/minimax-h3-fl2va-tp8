"""Run bounded real eight-DCU tests; never submit jobs or retry failed models."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE/'runtime'))
from h3_eightcard_evidence import REQUIRED_TESTS,runtime_identity,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--test-timeout-seconds',type=int,default=600)
    a=p.parse_args()
    if a.test_timeout_seconds<30:p.error('Timeout must be at least 30 seconds')
    output=a.output_dir.expanduser().resolve();output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    state=dict(status='RUNNING',started_at=datetime.now().astimezone().isoformat(),
               full_fl2va_tested=False,official_similarity_validated=False,test_reports={})
    def save():(output/'qualification-state.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
    save()
    try:
        identity=runtime_identity(PACKAGE);state['runtime_identity']=identity;save()
        child_env=dict(os.environ,H3_PLATFORM_QUALIFICATION_MODE='component-tests',
                       NINETOOTHED_DISABLE_FP_FUSION='1',PYTHONUNBUFFERED='1')
        test=PACKAGE/'tests/eight_card_components.py'
        state['test_program']=dict(path=str(test),sha256=sha(test))
        for name in ('collectives','native_eight_devices','tiny_tp8_h3','tiny_tp8_qwen'):
            report=output/(name+'.json');log=output/(name+'.log')
            command=[sys.executable,'-u']
            if name=='collectives':
                command += ['-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=8']
            command += [str(test),'--test',name,'--output',str(report)]
            state['stage']=name;save();print('EIGHT_CARD_COMPONENT_START',name,flush=True)
            test_started=time.monotonic()
            # A failed/timeout process group is terminated. There is no retry.
            # The allocation's platform deadline remains the final process-tree
            # cleanup boundary in case a native runtime cannot exit normally.
            with log.open('w',encoding='utf-8') as stream:
                process=subprocess.Popen(command,env=child_env,cwd=PACKAGE,
                    stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                try:returncode=process.wait(timeout=a.test_timeout_seconds)
                except subprocess.TimeoutExpired:
                    import signal
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
                    raise RuntimeError(name+' timed out; process group terminated, no automatic retry')
                except BaseException:
                    import signal
                    if process.poll() is None:
                        try:os.killpg(process.pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                        process.wait(timeout=10)
                    raise
            if returncode:raise RuntimeError(f'{name} failed with exit {returncode}; see {log}')
            data=json.loads(report.read_text())
            if data.get('status')!='PASS' or data.get('world_size')!=8:
                raise RuntimeError('Missing eight-device PASS evidence: '+name)
            state['test_reports'][name]=dict(path=str(report),sha256=sha(report),
                wall_seconds=time.monotonic()-test_started)
            save();print('EIGHT_CARD_COMPONENT_PASS',name,flush=True)
        if set(state['test_reports'])!=REQUIRED_TESTS:raise RuntimeError('Incomplete component set')
        if runtime_identity(PACKAGE)!=identity:raise RuntimeError('Runtime changed during component tests')
        state.update(status='EIGHT_CARD_COMPONENTS_PASS',stage='COMPLETE',
                     wall_seconds=time.monotonic()-started,finished_at=datetime.now().astimezone().isoformat())
        qualification=output/'eight-card-qualification.json'
        qualification.write_text(json.dumps(state,indent=2),encoding='utf-8');save()
        print('EIGHT_CARD_COMPONENT_QUALIFICATION_PASS',str(qualification),flush=True)
    except BaseException as error:
        state.update(status='FAIL',error=type(error).__name__+': '+str(error),wall_seconds=time.monotonic()-started)
        save();raise


if __name__=='__main__':main()
