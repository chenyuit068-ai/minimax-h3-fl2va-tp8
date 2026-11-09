"""Wait for the worker and record native signals without fabricating completion."""
import json,os,subprocess,time,uuid
from pathlib import Path

def supervise(command,audit_path):
    invocation=uuid.uuid4().hex
    environment=dict(os.environ,H3_WRAPPER_CHILD='1',H3_INVOCATION_ID=invocation)
    started=time.perf_counter()
    process=subprocess.run(command,env=environment,check=False)
    returncode=process.returncode if process.returncode>=0 else 128-process.returncode
    path=Path(audit_path) if audit_path is not None else None
    if path is not None and path.is_file():
        audit=json.loads(path.read_text())
        # Never modify an audit from an earlier invocation, even after a failed
        # attempt to reuse its output name.
        if audit.get('invocation_id')==invocation:
            audit.update(process_returncode=process.returncode,process_wall_seconds=time.perf_counter()-started)
            if audit.get('status')=='RUNNING':
                audit.update(status='FAILED_PROCESS_TERMINATION',full_fl2va_tested=False,competition_passed=False,
                    error=f'Worker terminated with return code {process.returncode}; see stderr/native traceback')
                if returncode==0:returncode=1
            path.write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    return returncode
