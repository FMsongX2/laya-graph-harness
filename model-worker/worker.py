"""Local JSONL worker; uses external candidate IDs, not fixed label classes."""
import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'vendor/laya'))
from build_data import compact

p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--head')
p.add_argument('--relative-temperature',type=float,default=1)
p.add_argument('--device',default='auto',help="auto = CUDA, else MPS, else CPU (Laya's own order)")
# The head temperature was calibrated from fp32 one-row forwards (MPS keeps a single row in fp32);
# CUDA would otherwise autocast to bf16/fp16 and shift the calibrated probabilities.
p.add_argument('--precision',choices=['fp32','amp'],default='fp32');args=p.parse_args()
# JSONL protocol is UTF-8 regardless of the Windows console code page.
sys.stdin.reconfigure(encoding='utf-8');sys.stdout.reconfigure(encoding='utf-8')
with contextlib.redirect_stdout(sys.stderr):
    import torch
    import laya
    from safetensors.torch import load_file
    torch.set_num_threads(4)
    agent=laya.load(args.model,device=None if args.device=='auto' else args.device)
    if args.precision=='fp32':
        agent.amp_enabled=False;agent.dtype=torch.float32
    agent.cfg.update(max_len=1536,head_max_len=512)
    if args.head:
        head=Path(args.head);config=json.loads((head/'head-config.json').read_text())
        missing,unexpected=agent.model.load_state_dict(load_file(str(head/'head.safetensors')),strict=False)
        assert not unexpected and all(k.startswith('encoder.') for k in missing)
        agent.temperature=[config['temperature'],1,1]
        agent.temperature_by_options={}
        agent.cfg.update(temperature=agent.temperature,temperature_by_options={})
    if args.relative_temperature!=1:
        agent.temperature=[t*args.relative_temperature for t in agent.temperature]
        agent.temperature_by_options={}
    agent.model.eval()
    captured=[]
    original_forward=agent._forward
    def capture_forward(batch):
        result=original_forward(batch);captured.append(result[0].tolist());return result
    agent._forward=capture_forward
print(json.dumps({'ready':True,'backend':'laya','device':str(agent.device),'device_type':agent.device.type,
                  'precision':str(agent.dtype_for(1)).removeprefix('torch.'),
                  'gpu':torch.cuda.get_device_name(agent.device) if agent.device.type=='cuda' else None}),flush=True)
for line in sys.stdin:
    try:
        request=compact(json.loads(line));started=time.perf_counter()
        captured.clear()
        with contextlib.redirect_stdout(sys.stderr):
            response=agent.predict(request['state'],request['questions'])
        response['latency_seconds']=time.perf_counter()-started
        if len(captured)==1 and len(captured[0])==len(request['questions']):
            response['raw_logits']={key:captured[0][i][:len(q['criteria'])]
                for i,(key,q) in enumerate(request['questions'].items())}
        for answer in response['answers'].values():
            if answer['type']=='choice':answer['confidence']=max(answer['probabilities'].values())
    except Exception as exc:
        response={'error':type(exc).__name__+': '+str(exc)}
    print(json.dumps(response,ensure_ascii=False,allow_nan=False),flush=True)
