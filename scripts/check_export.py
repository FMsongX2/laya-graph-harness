"""Check committed/exportable files only; never read ignored local asset contents."""
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
FORBIDDEN_NAMES={'auth.json','neo4j-secret.json','settings.local.json','decision.local.json'}
FORBIDDEN_PARTS={'.venv','.venv-model','__pycache__','vendor','base-model','trained','models','state','data','backups','logs'}
PATTERNS=[re.compile(r'/Users/[A-Za-z0-9_.-]+/'),re.compile(r'/home/[A-Za-z0-9_.-]+/'),
          re.compile(r'gh[pousr]_[A-Za-z0-9]{20,}'),re.compile(r'sk-[A-Za-z0-9_-]{24,}'),
          re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')]

def candidates():
    if (ROOT/'.git').exists():
        result=subprocess.run(['git','ls-files','-z'],cwd=ROOT,capture_output=True,check=True)
        return [ROOT/p for p in result.stdout.decode().split('\0') if p]
    return [p for p in ROOT.rglob('*') if p.is_file() and not any(x in FORBIDDEN_PARTS for x in p.relative_to(ROOT).parts)
            and p.name not in ['settings.json','decision.json']]

def check():
    files=candidates();errors=[]
    for p in files:
        relative=p.relative_to(ROOT)
        if p.is_symlink() or p.name in FORBIDDEN_NAMES or any(x in FORBIDDEN_PARTS for x in relative.parts):
            errors.append(str(relative)+': private/runtime asset');continue
        if p.stat().st_size>2_000_000:errors.append(str(relative)+': unexpected large file');continue
        try:content=p.read_text()
        except UnicodeDecodeError:errors.append(str(relative)+': unexpected binary');continue
        if any(pattern.search(content) for pattern in PATTERNS):errors.append(str(relative)+': host path or credential pattern')
    if errors:raise ValueError('\n'.join(errors))
    return {'passed':True,'files':len(files),'checked':'tracked or explicitly exportable source files only'}

if __name__=='__main__':print(json.dumps(check(),indent=2))
