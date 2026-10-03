"""One-command installer and asset check for a native (or container) deployment.

  install      create both environments, fetch pinned Laya, write ignored configs,
               build the Neo4j image, then report what private assets are missing
  check        report the deployment layout only; exit 1 when a required item is missing
  neo4j-image  build the configured Neo4j/APOC image if Docker lacks it

Standard library only. Never starts services, copies corpus data or downloads weights.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
KG = ROOT / 'knowledgegraph'
WINDOWS = os.name == 'nt'
TORCH_INDEX = 'https://download.pytorch.org/whl/'


def venv_python(venv):
    return venv / ('Scripts/python.exe' if WINDOWS else 'bin/python')


def say(status, message):
    print(f'[{status}] {message}', flush=True)


def accelerator(requested):
    if requested != 'auto': return requested
    if shutil.which('nvidia-smi') and subprocess.run(['nvidia-smi', '-L'], capture_output=True).returncode == 0:
        return 'cuda'
    if sys.platform == 'darwin' and platform.machine() == 'arm64': return 'mps'
    return 'cpu'


def torch_flavor(device):
    """PyTorch wheel index per device. cu128 includes Blackwell (sm_120) kernels; macOS uses PyPI."""
    return {'cuda': 'cu128', 'cpu': None if sys.platform == 'darwin' else 'cpu', 'mps': None}[device]


def pinned_torch():
    for line in (ROOT / 'requirements-model.txt').read_text(encoding='utf-8').splitlines():
        if line.startswith('torch=='): return line.split('==', 1)[1].strip()
    raise RuntimeError('requirements-model.txt does not pin torch')


def long_paths_enabled():
    if not WINDOWS: return True
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SYSTEM\CurrentControlSet\Control\FileSystem') as key:
            return winreg.QueryValueEx(key, 'LongPathsEnabled')[0] == 1
    except OSError:
        return False


def environment(venv, requirements, flavor, python, dry_run):
    """Create/update one venv; skipped when its requirement inputs are unchanged."""
    files = [ROOT / requirements, *sorted(ROOT.glob('requirements-test.txt'))]
    stamp = hashlib.sha256(json.dumps([flavor, *[f.read_text(encoding='utf-8') for f in files]]).encode()).hexdigest()
    marker = venv / '.laya-install.json'
    if marker.exists() and json.loads(marker.read_text(encoding='utf-8')).get('stamp') == stamp:
        say('ok', f'{venv.relative_to(ROOT)} is up to date')
        return
    say('..', f'installing {requirements} into {venv.relative_to(ROOT)} (torch: {flavor or "PyPI"})')
    if dry_run: return
    if not venv_python(venv).exists():
        subprocess.run([python, '-m', 'venv', str(venv)], check=True)
    pip = [str(venv_python(venv)), '-m', 'pip', 'install', '--disable-pip-version-check']
    if flavor:
        # Pin the accelerator build first; the plain `torch==X` requirement is then already satisfied.
        subprocess.run([*pip, '--extra-index-url', TORCH_INDEX + flavor, f'torch=={pinned_torch()}+{flavor}'], check=True)
    subprocess.run([*pip, '-r', str(ROOT / requirements)], check=True)
    marker.write_text(json.dumps({'stamp': stamp, 'torch_flavor': flavor}), encoding='utf-8')


def load(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def docker_ready():
    docker = shutil.which('docker')
    return bool(docker) and subprocess.run([docker, 'info'], capture_output=True, timeout=30).returncode == 0


def neo4j_image(dry_run=False):
    settings = load(KG / 'config/settings.json') or load(KG / 'config/settings.example.json')
    image = settings['graph']['image']
    if not docker_ready():
        say('skip', f'Docker is not running; build {image} later with: python scripts/install.py neo4j-image')
        return False
    if subprocess.run(['docker', 'image', 'inspect', image], capture_output=True).returncode == 0:
        say('ok', f'Neo4j image {image} present')
        return True
    say('..', f'building Neo4j image {image}')
    if not dry_run:
        subprocess.run(['docker', 'build', '-t', image, str(KG / 'config/neo4j')], check=True)
    return True


def check():
    """Every runtime prerequisite with a hint; returns False when a required item is missing."""
    settings, decision = load(KG / 'config/settings.json'), load(KG / 'config/decision.json')
    items = [('required', 'runtime environment', venv_python(KG / '.venv').exists(), 'python scripts/install.py'),
             ('required', 'settings.json', settings is not None, 'python scripts/configure.py, then edit it'),
             ('required', 'decision.json', decision is not None, 'python scripts/configure.py')]
    if decision:
        python = KG / decision['worker_python']
        if WINDOWS and python.parent.name == 'bin': python = python.parent.parent / 'Scripts/python.exe'
        backend=decision.get('backend','laya')
        items += [('required','model environment',python.exists(),
                   'see docs/decision2.md' if backend=='decision2' else 'python scripts/install.py')]
        if backend=='decision2':
            package=KG/(decision.get('decision2') or {}).get('model','missing-package')
            items += [('required','Decision 2.0 package',(package/'MODEL_MANIFEST.json').is_file(),
                       'private trusted package: see docs/decision2.md')]
        elif backend=='laya':
            lab = (KG / decision['model_lab']).resolve()
            head = lab / decision['adapter']
            items += [('required', 'pinned Laya source', (lab / 'vendor/laya/laya').is_dir(), 'python scripts/fetch_laya.py'),
                  ('required', 'Laya base model', (lab / 'base-model/model.safetensors').exists(),
                   'private asset: model-worker/base-model (see docs/deployment.md)'),
                  ('required', 'trained Laya head', (head / 'head.safetensors').exists() and (head / 'head-config.json').exists(),
                   'private asset: model-worker/trained/head.safetensors + head-config.json')]
        else:items += [('required','supported model backend',False,'use laya or decision2')]
    if settings:
        snapshot = Path(settings['embedding']['snapshot'])
        snapshot = snapshot if snapshot.is_absolute() else KG / snapshot
        items += [('required', 'BGE-M3 snapshot', snapshot.is_dir(), f'private asset: {settings["embedding"]["snapshot"]}'),
                  ('required', 'Neo4j encryption key', (KG / settings['graph']['encryption_key_file']).exists(),
                   f'private asset: {settings["graph"]["encryption_key_file"]}'),
                  ('required', 'dataset registry', (KG / 'state/system/databases/cognee_db').exists(),
                   'private state: knowledgegraph/state'),
                  ('required', 'approved source records', (KG / 'data/semantic-assertions').is_dir(),
                   'private data: knowledgegraph/data/semantic-assertions'),
                  ('required', 'Docker daemon', docker_ready(), 'start Docker Desktop / the Docker service')]
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(settings['chat']['upstream'].rstrip('/') + '/api/tags', timeout=3) as response:
                models = {m['name'] for m in json.load(response).get('models', [])}
            items.append(('optional', f'Ollama model {settings["chat"]["model"]}', settings['chat']['model'] in models,
                          f'ollama pull {settings["chat"]["model"]} (needed for search/build, not for select)'))
        except (OSError, ValueError):
            items.append(('optional', 'Ollama', False, f'not reachable at {settings["chat"]["upstream"]} (needed for search/build)'))
    ready = True
    for level, name, ok, hint in items:
        say('ok' if ok else ('missing' if level == 'required' else 'warn'), name + ('' if ok else f' -> {hint}'))
        ready &= ok or level != 'required'
    say('ok' if ready else 'not ready', 'deployment ' + ('is ready: kg prepare' if ready else 'needs the items above'))
    return ready


def install(args):
    device = accelerator(args.accelerator)
    say('ok', f'platform {sys.platform}/{platform.machine()}, accelerator {device}, Python {platform.python_version()}')
    if sys.version_info[:2] != (3, 12):
        say('warn', 'tested with Python 3.12; pass --python to choose an interpreter')
    if WINDOWS and device == 'cuda' and not long_paths_enabled() and len(str(ROOT)) > 60:
        raise SystemExit(f'Project path is {len(str(ROOT))} characters and Windows long paths are disabled; '
                         'CUDA torch headers would exceed 260 characters. Move the project or enable LongPathsEnabled.')
    flavor = torch_flavor(device)
    environment(KG / '.venv', 'requirements-runtime.txt', flavor, args.python, args.dry_run)
    decision=load(KG/'config/decision.json') or {}
    if not args.skip_model and decision.get('backend','laya')=='laya':
        environment(ROOT / '.venv-model', 'requirements-model.txt', flavor, args.python, args.dry_run)
        if not args.dry_run:
            sys.path.insert(0, str(ROOT / 'scripts'))
            import fetch_laya
            say('ok', f'Laya source at {fetch_laya.fetch().relative_to(ROOT)}')
    elif decision.get('backend')=='decision2':
        say('skip','Decision 2.0 environment/package provisioning follows docs/decision2.md; Laya assets are not required')
    if not args.dry_run:
        sys.path.insert(0, str(ROOT / 'scripts'))
        import configure
        for created in configure.configure(): say('ok', f'created {Path(created).relative_to(ROOT)} from example')
    if not args.skip_neo4j_image: neo4j_image(args.dry_run)
    print()
    return check()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', nargs='?', default='install', choices=['install', 'check', 'neo4j-image'])
    parser.add_argument('--accelerator', choices=['auto', 'cuda', 'cpu', 'mps'], default='auto')
    parser.add_argument('--python', default=sys.executable, help='interpreter used to create the environments')
    parser.add_argument('--skip-model', action='store_true', help='runtime environment only')
    parser.add_argument('--skip-neo4j-image', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='print the plan without installing')
    args = parser.parse_args()
    if args.command == 'check': raise SystemExit(0 if check() else 1)
    if args.command == 'neo4j-image': raise SystemExit(0 if neo4j_image(args.dry_run) else 1)
    raise SystemExit(0 if install(args) else 1)


if __name__ == '__main__':
    main()
