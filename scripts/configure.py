"""Generate ignored example configs for tests or local configuration; no service startup."""
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def configure(destination=None):
    root=Path(destination) if destination else ROOT/'knowledgegraph'
    config=root/'config';config.mkdir(parents=True,exist_ok=True)
    created=[]
    for name in ['settings','decision']:
        target=config/(name+'.json')
        if not target.exists():
            data=json.loads((ROOT/'knowledgegraph/config'/(name+'.example.json')).read_text())
            target.write_text(json.dumps(data,indent=2)+'\n');created.append(str(target))
    return created

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--destination');args=p.parse_args()
    print(json.dumps({'created':configure(args.destination),'services_started':False}))
