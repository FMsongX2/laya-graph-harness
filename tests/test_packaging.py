import importlib.util
import json
from pathlib import Path
import socket
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module

installer=load('overlay_installer',ROOT/'scripts/install_overlay.py')
bootstrap=load('portable_bootstrap',ROOT/'plugins/knowledgegraph-laya/skills/bounded-selection/scripts/bootstrap.py')

def unused_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));return sock.getsockname()[1]

class OverlayContract(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.target=Path(self.temp.name)
        for path in ['config','runtime','data','state']:(self.target/path).mkdir()
        self.settings={'server':{'port':unused_port()}}
        self.decision={'port':unused_port()}
        (self.target/'config/settings.json').write_text(json.dumps(self.settings))
        (self.target/'config/decision.json').write_text(json.dumps(self.decision))
        (self.target/'config/neo4j-secret.json').write_text('private sentinel')
        (self.target/'data/keep.txt').write_text('private data sentinel')
        (self.target/'runtime/decision_engine.py').write_text('original code sentinel')
    def tearDown(self):self.temp.cleanup()
    def test_preview_has_no_changes(self):
        r=installer.install(self.target)
        self.assertFalse(r['apply']);self.assertEqual((self.target/'runtime/decision_engine.py').read_text(),'original code sentinel')
        self.assertFalse((self.target/'backups').exists())
    def test_apply_preserves_private_assets_and_backups_code(self):
        r=installer.install(self.target,True)
        self.assertEqual((self.target/'config/neo4j-secret.json').read_text(),'private sentinel')
        self.assertEqual((self.target/'data/keep.txt').read_text(),'private data sentinel')
        self.assertEqual(json.loads((self.target/'config/settings.json').read_text()),self.settings)
        self.assertEqual((Path(r['backup'])/'runtime/decision_engine.py').read_text(),'original code sentinel')
        self.assertFalse(r['services_started'])
    def test_occupied_port_refuses_installation(self):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',self.settings['server']['port']));sock.listen()
            with self.assertRaises(RuntimeError):installer.install(self.target,True)
        self.assertEqual((self.target/'runtime/decision_engine.py').read_text(),'original code sentinel')
    def test_active_lease_refuses_installation(self):
        directory=self.target/'runtime/lifecycle';directory.mkdir()
        with installer.portable.locked(directory/'activity.lock',shared=True):
            with self.assertRaises(RuntimeError):installer.install(self.target,True)
        self.assertEqual((self.target/'runtime/decision_engine.py').read_text(),'original code sentinel')

class BootstrapContract(unittest.TestCase):
    def test_approved_skill_names_match(self):
        for name in ['domain-alignment','bounded-selection','knowledgegraph-laya:bounded-selection']:
            self.assertTrue(bootstrap.should_prepare({'hook_event_name':'UserPromptSubmit','prompt':'Use $'+name}))
    def test_other_skills_do_not_start_runtime(self):
        for text in ['ordinary prompt','$bounded-selection-extra','$debugging']:
            self.assertFalse(bootstrap.should_prepare({'hook_event_name':'UserPromptSubmit','prompt':text}))
    def test_source_reader_matches_new_skill_path(self):
        self.assertTrue(bootstrap.should_prepare({'hook_event_name':'PreToolUse','tool_name':'Read',
                         'tool_input':{'path':'/tmp/skills/bounded-selection/SKILL.md'}}))

if __name__=='__main__':unittest.main()
