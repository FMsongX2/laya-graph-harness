"""Managed resident selection service; standard-library client for fast agent calls."""
import json
from pathlib import Path
import socket
import subprocess
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
from runtime import portable
CONFIG = json.loads((ROOT/'config/decision.json').read_text())
BASE = f"http://127.0.0.1:{CONFIG['port']}"
STATE = ROOT/'runtime/decision-process.json'
# Model load and the first CUDA forward each get the worker's load allowance.
WARMUP_TIMEOUT = 40 + CONFIG.get('worker_load_timeout_seconds', 0)


def call(path, payload=None, timeout=210):
    body = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE + path, data=body, headers={'Content-Type': 'application/json'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout) as r: return json.load(r)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f'HTTP {exc.code}: ' + exc.read().decode(errors='replace')) from None
    except urllib.error.URLError:
        raise RuntimeError('Local selection service is unavailable; run ./kg decision-up') from None


def owned():
    if not STATE.exists(): return None
    state = json.loads(STATE.read_text())
    cmd = portable.command_line(state['pid'])
    return state if 'decision_service:app' in cmd and str(ROOT/'runtime') in cmd else None


def status():
    state = owned()
    return {'running': bool(state), 'process': state, 'health': call('/health', timeout=3) if state else None}


def start():
    from runtime.environment import environment
    lock_path = ROOT/'runtime/decision-start.lock'
    with portable.locked(lock_path):
        state = owned()
        if state:
            try:
                health = call('/health', timeout=3)
                if not health['ready']: health = call('/warmup', {}, timeout=WARMUP_TIMEOUT)
                if health['ready']: return {'running': True, 'pid': state['pid'], 'health': health}
            except RuntimeError: pass
            # A dead worker is handled by Engine on its next request; do not kill active work.
        else:
            with socket.socket() as probe:
                if probe.connect_ex(('127.0.0.1', CONFIG['port'])) == 0:
                    raise RuntimeError('Selection port is occupied by another process')
            command = [str(portable.venv_python(ROOT/'.venv')), '-m', 'uvicorn', 'decision_service:app',
                       '--app-dir', str(ROOT/'runtime'), '--host', '127.0.0.1', '--port', str(CONFIG['port']),
                       '--workers', '1', '--no-access-log']
            log_path = ROOT/'logs/decision-service.log'; log_path.parent.mkdir(exist_ok=True)
            with log_path.open('a') as log:
                child = subprocess.Popen(command, cwd=ROOT, env=environment(), stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=log, **portable.detached())
            state = {'pid': child.pid, 'command': command, 'log': str(log_path), 'started_at': time.time()}
            STATE.write_text(json.dumps(state, indent=2) + '\n')
        deadline = time.monotonic() + WARMUP_TIMEOUT
        while time.monotonic() < deadline:
            if not owned(): raise RuntimeError('Selection service exited; inspect logs/decision-service.log')
            try:
                health = call('/health', timeout=2)
                if health['ready']: return {'running': True, 'pid': state['pid'], 'health': health}
            except RuntimeError: pass
            time.sleep(.1)
        raise RuntimeError('Selection service did not become ready; inspect logs/decision-service.log')


def stop():
    state = owned()
    if not state: return {'running': False}
    portable.request_stop(state['pid'])
    for _ in range(60):
        if not owned(): return {'running': False, 'stopped_pid': state['pid']}
        time.sleep(.1)
    raise RuntimeError('Selection service is finishing active work; no force kill performed')


def selection(payload):
    try:
        health = call('/health', timeout=2)
    except RuntimeError:
        start()
    else:
        if health.get('service') != 'knowledgegraph-decision':
            raise RuntimeError('Selection port belongs to another service')
    return call('/select-batch' if 'items' in payload else '/select', payload)
