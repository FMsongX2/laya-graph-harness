"""Ownership and activity boundaries for cleanup, without modifying real datasets."""
import asyncio
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime import lifecycle


class LeaseContract(unittest.TestCase):
    def test_cli_help_does_not_acquire_a_lease_or_start_a_watcher(self):
        for command in ['select','walk','prepare','decision-down']:
            with patch.object(lifecycle.sys,'argv',['kg',command,'--help']), \
                 patch.object(lifecycle,'Lease') as lease,patch.object(lifecycle,'ensure_watcher') as watcher:
                self.assertEqual(lifecycle.tracked_cli(lambda:'help')(),'help')
                lease.assert_not_called();watcher.assert_not_called()

    def test_live_lease_prevents_cleanup_even_at_expired_deadline(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(lifecycle,'LOCK',Path(d)/'lock'), patch.object(lifecycle,'LAST_USE',Path(d)/'last'):
                with lifecycle.Lease():
                    result=lifecycle.cleanup_if_idle(idle_timeout=0)
                    self.assertEqual(result['reason'],'active_lease')
                self.assertTrue(lifecycle.LAST_USE.exists())

    def test_asgi_holds_lease_through_the_actual_request(self):
        async def inner(scope,receive,send):
            self.assertEqual(lifecycle.cleanup_if_idle(idle_timeout=0)['reason'],'active_lease')
        with tempfile.TemporaryDirectory() as d:
            with patch.object(lifecycle,'LOCK',Path(d)/'lock'), patch.object(lifecycle,'LAST_USE',Path(d)/'last'):
                asyncio.run(lifecycle.ActivityMiddleware(inner)({'type':'http','path':'/select'},None,None))

    def test_health_poll_does_not_refresh_idle_timer(self):
        async def inner(scope,receive,send): pass
        with tempfile.TemporaryDirectory() as d:
            with patch.object(lifecycle,'LOCK',Path(d)/'lock'), patch.object(lifecycle,'LAST_USE',Path(d)/'last'):
                asyncio.run(lifecycle.ActivityMiddleware(inner)({'type':'http','path':'/health'},None,None))
                self.assertFalse(lifecycle.LAST_USE.exists())

    def test_live_ingestion_worker_defers_cleanup(self):
        from tools import cli
        with tempfile.TemporaryDirectory() as d:
            with patch.object(lifecycle,'LOCK',Path(d)/'lock'), patch.object(lifecycle,'LAST_USE',Path(d)/'last'), \
                 patch.object(cli,'paper_status',return_value={'worker_running':True}):
                result=lifecycle.cleanup_if_idle(idle_timeout=0)
                self.assertFalse(result['cleaned'])
                self.assertEqual(result['reason'],'ingestion_worker_active')

    def test_draining_request_is_rejected_without_entering_model(self):
        entered=[];messages=[]
        async def inner(scope,receive,send):entered.append(True)
        async def send(message):messages.append(message)
        with tempfile.TemporaryDirectory() as d:
            with patch.object(lifecycle,'LOCK',Path(d)/'lock'), patch.object(lifecycle,'LAST_USE',Path(d)/'last'):
                with lifecycle.LOCK.open('a') as lock:
                    lifecycle.fcntl.flock(lock,lifecycle.fcntl.LOCK_EX)
                    asyncio.run(lifecycle.ActivityMiddleware(inner)({'type':'http','path':'/select'},None,send))
                self.assertFalse(entered)
                self.assertEqual(messages[0]['status'],503)

    def test_shutdown_owner_mismatch_refuses_force_kill(self):
        def fail(): raise RuntimeError('graceful shutdown failed')
        with patch.object(lifecycle.os,'killpg') as kill:
            with self.assertRaises(RuntimeError):
                lifecycle.stop_idle_service(fail,lambda:{'pid':os.getpid()},{'pid':os.getpid()+1})
            kill.assert_not_called()

    def test_idle_owned_group_can_be_released_after_grace_failure(self):
        process=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True)
        def fail():raise RuntimeError('controlled grace failure')
        def owned():return {'pid':process.pid} if process.poll() is None else None
        try:
            self.assertTrue(lifecycle.stop_idle_service(fail,owned,{'pid':process.pid}))
            self.assertEqual(process.wait(timeout=5),-9)
        finally:
            if process.poll() is None:process.terminate();process.wait(timeout=5)


if __name__=='__main__':unittest.main()
