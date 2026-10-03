import importlib.util
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module

installer=load('native_installer',ROOT/'scripts/install.py')

def dockerignore_excludes(path):
    """Docker's rule: the last pattern matching the path or any parent decides."""
    def regex(pattern):
        out='';i=0
        while i<len(pattern):
            if pattern.startswith('**',i):out+='.*';i+=2
            elif pattern[i]=='*':out+='[^/]*';i+=1
            elif pattern[i]=='?':out+='[^/]';i+=1
            else:out+=re.escape(pattern[i]);i+=1
        return re.compile(out+'$')
    rules=[]
    for line in (ROOT/'.dockerignore').read_text(encoding='utf-8').splitlines():
        line=line.strip()
        if line and not line.startswith('#'):
            negated=line.startswith('!');rules.append((negated,regex(line.lstrip('!').rstrip('/'))))
    parts=path.split('/');candidates=['/'.join(parts[:i]) for i in range(1,len(parts)+1)]
    excluded=False
    for negated,pattern in rules:
        if any(pattern.match(c) for c in candidates):excluded=not negated
    return excluded

class ImageContextContract(unittest.TestCase):
    def test_private_assets_never_enter_the_build_context(self):
        for path in ['knowledgegraph/data/semantic-assertions/P01/P01-A001.json','knowledgegraph/state/system/databases/cognee_db',
                     'knowledgegraph/config/settings.json','knowledgegraph/config/decision.json',
                     'knowledgegraph/config/neo4j-secret.json','knowledgegraph/models/bge-m3/config.json',
                     'model-worker/trained/head.safetensors','model-worker/base-model/model.safetensors',
                     'model-worker/vendor/laya/setup.py','knowledgegraph/.venv/pyvenv.cfg','.venv-model/pyvenv.cfg',
                     'laya-home/config/settings.json','verification/local-results.json','.git/config',
                     'knowledgegraph/runtime/__pycache__/x.pyc','knowledgegraph/logs/service.log']:
            self.assertTrue(dockerignore_excludes(path),path)
    def test_authored_runtime_is_in_the_build_context(self):
        for path in ['Dockerfile','docker/entrypoint.sh','requirements-runtime.txt','scripts/install.py',
                     'scripts/fetch_laya.py','knowledgegraph/kg','knowledgegraph/runtime/decision_engine.py',
                     'knowledgegraph/tools/cli.py','knowledgegraph/config/settings.example.json',
                     'knowledgegraph/config/lifecycle.json','knowledgegraph/config/neo4j/Dockerfile','model-worker/worker.py']:
            self.assertFalse(dockerignore_excludes(path),path)
        for copied in re.findall(r'^COPY (?!--)(\S+)',(ROOT/'Dockerfile').read_text(encoding='utf-8'),re.M):
            if '*' not in copied and copied!='.':self.assertFalse(dockerignore_excludes(copied),copied)

class InstallerContract(unittest.TestCase):
    def test_torch_pin_follows_model_requirements(self):
        self.assertRegex(installer.pinned_torch(),r'^\d+\.\d+\.\d+$')
    def test_cuda_uses_blackwell_capable_wheels(self):
        self.assertEqual(installer.torch_flavor('cuda'),'cu128')
        self.assertIsNone(installer.torch_flavor('mps'))
    def test_check_reports_every_missing_private_asset(self):
        with tempfile.TemporaryDirectory() as d:
            kg=Path(d)/'knowledgegraph';(kg/'config').mkdir(parents=True)
            for name in ['settings','decision']:
                (kg/'config'/(name+'.json')).write_text((ROOT/'knowledgegraph/config'/(name+'.example.json')).read_text(encoding='utf-8'))
            lines=[]
            with patch.object(installer,'KG',kg),patch.object(installer,'docker_ready',return_value=False), \
                 patch.object(installer,'say',lambda status,message:lines.append((status,message))):
                self.assertFalse(installer.check())
            missing={message.split(' ->')[0] for status,message in lines if status=='missing'}
            for item in ['runtime environment','Laya base model','trained Laya head','BGE-M3 snapshot',
                         'Neo4j encryption key','dataset registry','approved source records','Docker daemon']:
                self.assertIn(item,missing)
            self.assertNotIn('settings.json',missing)
    def test_unchanged_environment_is_not_reinstalled(self):
        with tempfile.TemporaryDirectory() as d:
            venv=Path(d)/'.venv';venv.mkdir()
            with patch.object(installer,'ROOT',Path(d)),patch.object(installer,'say',lambda *a:None), \
                 patch.object(installer.subprocess,'run') as run:
                for name in ['requirements-runtime.txt','requirements-test.txt']:(Path(d)/name).write_text('x==1\n')
                stamp_files=[Path(d)/'requirements-runtime.txt',Path(d)/'requirements-test.txt']
                import hashlib
                stamp=hashlib.sha256(json.dumps(['cu128',*[f.read_text(encoding='utf-8') for f in stamp_files]]).encode()).hexdigest()
                (venv/'.laya-install.json').write_text(json.dumps({'stamp':stamp}))
                installer.environment(venv,'requirements-runtime.txt','cu128','python',False)
                run.assert_not_called()

if __name__=='__main__':unittest.main()
