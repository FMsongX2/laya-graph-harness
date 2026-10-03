"""Platform boundaries the runtime relies on: locks, graceful stop, and the worker pipe."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
import decision_engine
import portable


class LockContract(unittest.TestCase):
    def test_shared_leases_coexist_and_block_exclusive_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'activity.lock'
            with portable.locked(path, shared=True), portable.locked(path, shared=True, blocking=False):
                with self.assertRaises(BlockingIOError):
                    with portable.locked(path, blocking=False): pass
            with portable.locked(path, blocking=False): pass

    def test_exclusive_lock_blocks_another_process(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'start.lock'
            probe = ('import sys; sys.path.insert(0, sys.argv[2]); import portable\n'
                     'try:\n    with portable.locked(sys.argv[1], blocking=False): print("free")\n'
                     'except BlockingIOError: print("held")')
            run = lambda: subprocess.run([sys.executable, '-c', probe, str(path), str(Path(portable.__file__).parent)],
                                         capture_output=True, text=True, check=True).stdout.strip()
            with portable.locked(path):
                self.assertEqual(run(), 'held')
            self.assertEqual(run(), 'free')


class ProcessContract(unittest.TestCase):
    def test_request_stop_runs_the_services_own_shutdown(self):
        with tempfile.TemporaryDirectory() as d:
            marker, ready = Path(d) / 'stopped', Path(d) / 'ready'
            script = textwrap.dedent(f'''
                import pathlib, signal, sys, time
                def stop(*_):
                    pathlib.Path({str(marker)!r}).write_text('graceful'); sys.exit(0)
                for name in ('SIGTERM', 'SIGBREAK'):
                    if hasattr(signal, name): signal.signal(getattr(signal, name), stop)
                pathlib.Path({str(ready)!r}).write_text('ready')
                # Short ticks like a server loop: a Windows sleep wakes early only for SIGINT.
                while True: time.sleep(.05)''')
            child = subprocess.Popen([sys.executable, '-c', script], stdin=subprocess.DEVNULL, **portable.detached())
            try:
                deadline = time.monotonic() + 20
                while not ready.exists() and time.monotonic() < deadline: time.sleep(.05)
                self.assertIn('-c', portable.command_line(child.pid))
                portable.request_stop(child.pid)
                self.assertEqual(child.wait(timeout=10), 0)
                self.assertEqual(marker.read_text(), 'graceful')
                self.assertEqual(portable.command_line(child.pid), '')
            finally:
                if child.poll() is None: portable.kill_group(child.pid); child.wait(timeout=5)

    def test_kill_group_releases_the_owned_tree(self):
        script = 'import subprocess, sys, time; subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"]); time.sleep(60)'
        child = subprocess.Popen([sys.executable, '-c', script], **portable.detached())
        time.sleep(1)
        portable.kill_group(child.pid)
        self.assertIsNotNone(child.wait(timeout=10))

    def test_windows_venv_layout(self):
        configured = Path('../.venv-model/bin/python')
        expected = Path('../.venv-model/Scripts/python.exe') if portable.WINDOWS else configured
        self.assertEqual(portable.interpreter(configured), expected)
        self.assertEqual(portable.passthrough_environment([])['PYTHONUTF8'], '1')


FAKE_WORKER = r'''
import argparse, json, sys, time
p = argparse.ArgumentParser(); p.add_argument('--model'); p.add_argument('--head')
p.add_argument('--device'); p.add_argument('--precision'); args = p.parse_args()
sys.stdin.reconfigure(encoding='utf-8'); sys.stdout.reconfigure(encoding='utf-8')
print(json.dumps({'ready': True, 'device': 'cpu', 'device_type': 'cpu', 'args': [args.device, args.precision]}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if request.get('hang'): time.sleep(30)
    print(json.dumps({'echo': request['text']}, ensure_ascii=False), flush=True)
'''


class WorkerPipeContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); lab = Path(self.temp.name)
        (lab / 'worker.py').write_text(FAKE_WORKER, encoding='utf-8')
        (lab / 'base-model').mkdir(); (lab / 'trained').mkdir(); (lab / 'trained/head.safetensors').write_bytes(b'')
        self.config = patch.dict(decision_engine.CONFIG, {'model_lab': str(lab), 'worker_python': sys.executable,
                                 'adapter': 'trained', 'device': 'auto', 'precision': 'fp32',
                                 'worker_timeout_seconds': 1, 'worker_load_timeout_seconds': 20})
        self.config.start(); self.worker = decision_engine.Worker()

    def tearDown(self):
        self.worker.close(); self.config.stop(); self.temp.cleanup()

    def test_utf8_round_trip_through_the_resident_pipe(self):
        response, _ = self.worker.evaluate({'text': '근거 — evidence ✓'})
        self.assertEqual(response, {'echo': '근거 — evidence ✓'})
        self.assertEqual(self.worker.device, 'cpu')

    def test_timeout_discards_the_worker_and_restarts_lazily(self):
        self.worker.start(); first = self.worker.proc.pid
        with self.assertRaisesRegex(RuntimeError, 'timed out'):
            self.worker.evaluate({'hang': True, 'text': 'x'})
        self.assertIsNone(self.worker.proc)
        response, _ = self.worker.evaluate({'text': 'again'})
        self.assertEqual(response['echo'], 'again')
        self.assertNotEqual(self.worker.proc.pid, first)

    def test_configured_device_mismatch_is_refused(self):
        with patch.dict(decision_engine.CONFIG, {'device': 'cuda'}):
            with self.assertRaisesRegex(RuntimeError, 'configured device'):
                self.worker.start()
        self.assertIsNone(self.worker.proc)


if __name__ == '__main__':
    unittest.main()
