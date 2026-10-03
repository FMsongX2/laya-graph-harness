"""Install authored runtime code into an existing offline KnowledgeGraph project.

Dataset, secret, model, settings and runtime state files are never copied.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import socket
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'knowledgegraph'))
from runtime import portable

def source_files():
    kg=ROOT/'knowledgegraph'
    return [kg/'kg',kg/'kg.cmd',*sorted((kg/'runtime').glob('*.py')),*sorted((kg/'tools').glob('*.py'))]

def process_alive(path):
    if not path.exists():return False
    state=json.loads(path.read_text())
    return portable.command_line(state['pid'])!=''

def install(target,apply=False):
    target=Path(target).expanduser().resolve();kg=ROOT/'knowledgegraph'
    settings_path=target/'config/settings.json';decision_path=target/'config/decision.json'
    if not settings_path.exists() or not decision_path.exists():
        raise ValueError('Target must be an existing configured KnowledgeGraph project with decision.json')
    settings=json.loads(settings_path.read_text());decision=json.loads(decision_path.read_text())
    files=source_files();plan={'target':str(target),'files':[str(p.relative_to(kg)) for p in files],
        'settings_preserved':True,'dataset_and_weights_copied':False,'apply':apply}
    if not apply:return plan
    control=target/'runtime/lifecycle';control.mkdir(exist_ok=True)
    with (control/'activity.lock').open('a') as lease:
        try:portable.lock(lease,blocking=False)
        except BlockingIOError:raise RuntimeError('Target has active runtime work; stop it before installation') from None
        for state in [target/'runtime/service-process.json',target/'runtime/decision-process.json',control/'process.json']:
            if process_alive(state):raise RuntimeError('Stop target services and its idle watcher before installing the overlay')
        for port in [settings['server']['port'],decision['port']]:
            with socket.socket() as probe:
                if probe.connect_ex(('127.0.0.1',port))==0:raise RuntimeError('Target port is occupied; parallel deployment refused')
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        backup=target/'backups'/('laya-graph-harness-'+stamp);backup.mkdir(parents=True,exist_ok=False)
        manifest=[]
        for source in files:
            relative=source.relative_to(kg);dest=target/relative
            if dest.exists():
                saved=backup/relative;saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(dest,saved)
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest);manifest.append(str(relative))
        lifecycle=target/'config/lifecycle.json'
        if not lifecycle.exists():shutil.copy2(kg/'config/lifecycle.json',lifecycle)
        (backup/'manifest.json').write_text(json.dumps({'installed':manifest},indent=2))
        return {**plan,'backup':str(backup),'services_started':False}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--target',required=True)
    p.add_argument('--apply',action='store_true');args=p.parse_args()
    print(json.dumps(install(args.target,args.apply),indent=2))
