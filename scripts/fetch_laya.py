"""Fetch the pinned upstream Laya source; does not download weights or infer."""
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
URL='https://github.com/NandhaKishorM/laya.git'
REVISION='fa9a2a7070b1789912a49ae24603bbfb1a78b001'

def fetch():
    target=ROOT/'model-worker/vendor/laya'
    if target.exists():
        actual=subprocess.run(['git','-C',str(target),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
        if actual!=REVISION:raise RuntimeError('Existing vendor checkout differs; it was not overwritten')
        return target
    target.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(['git','clone','--filter=blob:none','--no-checkout',URL,str(target)],check=True)
    subprocess.run(['git','-C',str(target),'checkout','--detach',REVISION],check=True)
    return target

if __name__=='__main__':print(fetch())
