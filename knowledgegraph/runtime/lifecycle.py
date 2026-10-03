"""Shared activity leases and one idle reaper for this project's owned runtime."""
from __future__ import annotations
from contextlib import contextmanager
import functools
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import urllib.request
try: from . import portable
except ImportError: import portable

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT/'runtime/lifecycle'
CONTROL.mkdir(exist_ok=True)
LOCK = CONTROL/'activity.lock'
LAST_USE = CONTROL/'last-use'
STATE = CONTROL/'process.json'
REGISTRY = (ROOT/'state/system/databases/cognee_db').as_uri() + '?mode=ro'
CONFIG = json.loads((ROOT/'config/lifecycle.json').read_text())


class Lease:
    """Separate file descriptions make concurrent CLI/HTTP/thread readers visible."""
    def __init__(self, blocking=True): self.blocking = blocking; self.fd = None
    def __enter__(self):
        self.fd = LOCK.open('a')
        try:
            portable.lock(self.fd, shared=True, blocking=self.blocking)
        except BlockingIOError:
            self.fd.close(); self.fd = None
            raise RuntimeError('Runtime is being released; retry after cleanup') from None
        LAST_USE.touch()
        return self
    def __exit__(self, *exc):
        if self.fd:
            LAST_USE.touch(); portable.unlock(self.fd); self.fd.close(); self.fd = None


def tracked_operation(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with Lease(blocking=False): return fn(*args, **kwargs)
    return wrapper


def owned_watcher():
    if not STATE.exists(): return None
    try: state = json.loads(STATE.read_text())
    except (ValueError, OSError): return None
    cmd = portable.command_line(state['pid'])
    return state if str(ROOT/'runtime/lifecycle.py') in cmd and 'watch' in cmd else None


def ensure_watcher():
    with portable.locked(CONTROL/'start.lock'):
        state = owned_watcher()
        version = hashlib.sha256(Path(__file__).read_bytes() + (ROOT/'config/lifecycle.json').read_bytes()).hexdigest()
        if state and state.get('version') == version: return state
        if state:
            portable.request_stop(state['pid'])
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and owned_watcher(): time.sleep(.05)
            if owned_watcher(): raise RuntimeError('Previous idle watcher did not exit; duplicate watcher refused')
        (ROOT/'logs').mkdir(exist_ok=True)
        command = [str(portable.venv_python(ROOT/'.venv')), str(ROOT/'runtime/lifecycle.py'), 'watch']
        with (ROOT/'logs/lifecycle.log').open('a') as log:
            proc = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=log, **portable.detached())
        state = {'pid': proc.pid, 'command': command, 'started_at': time.time(), 'version': version}
        temporary = STATE.with_suffix('.tmp')
        temporary.write_text(json.dumps(state)); temporary.replace(STATE)
        return state


def tracked_cli(fn):
    @functools.wraps(fn)
    def wrapper():
        # Argparse help is metadata access, not managed runtime work.
        if any(arg in {'-h','--help'} for arg in sys.argv[1:]):
            return fn()
        command = sys.argv[1] if len(sys.argv) > 1 else ''
        active = {'up', 'prepare', 'select', 'walk', 'search', 'datasets', 'add', 'build',
                  'ingest-papers', 'decision-up'}
        if command in active:
            with Lease():
                ensure_watcher()
                return fn()
        if command in {'down', 'decision-down'}:
            with portable.locked(LOCK):
                return fn()
        return fn()
    return wrapper


class ActivityMiddleware:
    """Hold a lease through the full ASGI request, including queues and cancellation."""
    def __init__(self, app): self.app = app
    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope.get('path') in {'/health', '/health/ready', '/runtime/activity'}:
            return await self.app(scope, receive, send)
        lease = Lease(blocking=False)
        try: lease.__enter__()
        except RuntimeError:
            body = b'{"error":"runtime_draining","retryable":true}'
            await send({'type':'http.response.start','status':503,
                        'headers':[(b'content-type',b'application/json'),(b'content-length',str(len(body)).encode())]})
            return await send({'type':'http.response.body','body':body})
        try: return await self.app(scope, receive, send)
        finally: lease.__exit__()


def status():
    age = max(0, time.time() - LAST_USE.stat().st_mtime) if LAST_USE.exists() else None
    return {'watcher': owned_watcher(), 'idle_seconds': age,
            'idle_timeout_seconds': CONFIG['idle_timeout_seconds'], 'poll_seconds': CONFIG['poll_seconds']}


def health(port, path='/health'):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f'http://127.0.0.1:{port}{path}', timeout=3) as response:
        return json.load(response)


def registered_containers():
    """Only containers registered in this project's relational dataset registry."""
    with sqlite3.connect(REGISTRY, uri=True) as c:
        rows = c.execute("SELECT dataset_id,graph_database_connection_info FROM dataset_database WHERE graph_database_provider='neo4j' AND graph_dataset_database_handler='neo4j_community'").fetchall()
    result = []
    for dataset_id, raw in rows:
        name = json.loads(raw).get('container_name')
        if name == 'cognee-neo4j-' + str(dataset_id).replace('-', ''):
            result.append(name)
    return result


def foreign_transactions():
    """Do not close a graph while an unwrapped direct Neo4j transaction is active."""
    import base64
    import hashlib
    from cryptography.fernet import Fernet
    from neo4j import GraphDatabase
    settings = json.loads((ROOT/'config/settings.json').read_text())
    key = json.loads((ROOT/settings['graph']['encryption_key_file']).read_text())['encryption_key']
    cipher = Fernet(base64.urlsafe_b64encode(hashlib.sha256(key.encode()).digest()))
    with sqlite3.connect(REGISTRY, uri=True) as c:
        rows = c.execute("SELECT graph_database_url,graph_database_name,graph_database_connection_info FROM dataset_database WHERE graph_database_provider='neo4j' AND graph_dataset_database_handler='neo4j_community'").fetchall()
    owned_names = set(registered_containers())
    query = 'SHOW TRANSACTIONS YIELD currentQuery, transactionId WHERE currentQuery IS NULL OR currentQuery <> $probe RETURN transactionId'
    for uri, database, raw in rows:
        info = json.loads(raw);name = info['container_name']
        if name not in owned_names: continue
        running = subprocess.run(['docker','inspect','--format','{{.State.Running}}',name], capture_output=True, text=True, timeout=10)
        if running.returncode or running.stdout.strip() != 'true': continue
        password = cipher.decrypt(info['graph_database_password'].encode()).decode()
        with GraphDatabase.driver(uri, auth=(info['graph_database_username'],password), connection_timeout=3) as driver:
            with driver.session(database=database, default_access_mode='READ') as session:
                if session.run(query, probe=query).data(): return True
    return False


def stop_idle_service(stop_fn, owned_fn, state):
    """Escalate only a verified owned process group after idle graceful shutdown fails."""
    try:
        stop_fn()
        return False
    except RuntimeError:
        current = owned_fn()
        if not current: return False
        if current['pid'] != state['pid'] or not portable.is_group_leader(current['pid']):
            raise RuntimeError('Idle shutdown ownership changed; force shutdown refused') from None
        portable.kill_group(current['pid'])
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if not owned_fn(): return True
            time.sleep(.05)
        raise RuntimeError('Owned idle process did not exit after shutdown')


def cleanup_if_idle(*, idle_timeout=None, retire=False):
    timeout = CONFIG['idle_timeout_seconds'] if idle_timeout is None else idle_timeout
    with LOCK.open('a') as lock:
        try: portable.lock(lock, blocking=False)
        except BlockingIOError: return {'cleaned':False,'reason':'active_lease'}
        if LAST_USE.exists() and time.time() - LAST_USE.stat().st_mtime < timeout:
            return {'cleaned':False,'reason':'recent_use'}
        sys.path.insert(0,str(ROOT))
        from tools import cli, decision_cli
        if cli.paper_status()['worker_running']:
            return {'cleaned':False,'reason':'ingestion_worker_active'}
        native = cli.owned_process(); decision = decision_cli.owned()
        # A partial restart or unknown health state is not proof that a service is idle.
        for state, port, path in [(native, cli.CONFIG['server']['port'], '/runtime/activity'),
                                  (decision, decision_cli.CONFIG['port'], '/health')]:
            if not state: continue
            try: value = health(port, path)
            except Exception: return {'cleaned':False,'reason':'service_health_unavailable'}
            if (value.get('busy') or value.get('chat_busy') or value.get('embedding_busy') or
                    value.get('knowledge_operation')):
                return {'cleaned':False,'reason':'service_work_active'}
        if CONFIG['stop_owned_neo4j_containers'] and foreign_transactions():
            return {'cleaned':False,'reason':'neo4j_transaction_active'}
        stopped = []; forced = []
        if decision:
            if stop_idle_service(decision_cli.stop, decision_cli.owned, decision): forced.append('laya_service')
            stopped.append('laya_service')
        if native:
            if stop_idle_service(cli.stop, cli.owned_process, native): forced.append('cognee_embedding_service')
            stopped.append('cognee_embedding_service')
        containers = []
        if CONFIG['stop_owned_neo4j_containers']:
            for name in registered_containers():
                p = subprocess.run(['docker','inspect','--format','{{.State.Running}}',name], capture_output=True,text=True,timeout=10)
                if p.returncode == 0 and p.stdout.strip() == 'true':
                    subprocess.run(['docker','stop','--time','30',name],capture_output=True,check=True,timeout=40)
                    containers.append(name)
        if retire and STATE.exists():
            # Remove this watcher before releasing the exclusive lease: the next start
            # must not reuse a process that is about to exit.
            state = json.loads(STATE.read_text())
            if state['pid'] == os.getpid(): STATE.unlink()
        result = {'cleaned':True,'stopped':stopped,'forced_owned_process_groups':forced,'neo4j_containers_stopped':containers,
                  'volumes_deleted':False,'shared_docker_or_ollama_stopped':False}
        with (ROOT/'logs/lifecycle-cleanups.jsonl').open('a') as log:
            log.write(json.dumps({'time':time.time(),**result})+'\n')
        return result


def watch():
    while True:
        time.sleep(CONFIG['poll_seconds'])
        try:
            if cleanup_if_idle(retire=True)['cleaned']: return
        except Exception as exc:
            # Operational errors remain visible; retry later instead of killing unknown work.
            print(json.dumps({'cleanup_failed':type(exc).__name__}),flush=True)


if __name__ == '__main__':
    if sys.argv[1:] != ['watch']: raise SystemExit('Use lifecycle.py watch')
    watch()
