"""Offline verified Decision 2.0 package behind the harness JSONL worker contract."""
import argparse
import contextlib
import importlib
import json
import os
from pathlib import Path
import re
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--mps-memory-fraction',type=float,default=.5)
    args=parser.parse_args()
    if not re.fullmatch(r'cpu|mps|cuda(?::\d+)?',args.device):parser.error('Use cpu, mps or cuda[:index]')
    if not 1<=args.threads<=64:parser.error('threads must be between 1 and 64')
    if not 0<args.mps_memory_fraction<=1:parser.error('mps-memory-fraction must be in (0,1]')
    package=Path(args.model).expanduser().resolve(strict=True)
    if not (package/'MODEL_MANIFEST.json').is_file():parser.error('Provide a complete verified Decision 2.0 package')
    # Set before importing torch. Unsupported MPS operators must fail explicitly.
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
                      PYTORCH_ENABLE_MPS_FALLBACK='0')
    started=time.perf_counter()
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        torch.set_num_threads(args.threads)
        if args.device=='mps':
            if not torch.backends.mps.is_available():raise RuntimeError('MPS is unavailable')
            torch.mps.set_per_process_memory_fraction(args.mps_memory_fraction)
        sys.path.insert(0,str(package))
        native=importlib.import_module('decision2')
        # The native loader checks package hashes, model identity and parameter count.
        engine=native.Decision2.from_pretrained(package,device=args.device,threads=args.threads,
            bf16_resident=args.device.startswith('cuda'),graphs=False,kernels=False)

    def synchronize():
        if args.device=='mps':torch.mps.synchronize()
        elif args.device.startswith('cuda'):torch.cuda.synchronize(args.device)

    synchronize()
    print(json.dumps({'ready':True,'backend':'decision2','model':engine.model_name,'device':args.device,
                      'load_seconds':time.perf_counter()-started,'cpu_fallback_enabled':False}),flush=True)
    for line in sys.stdin:
        try:
            payload=json.loads(line)
            synchronize();started=time.perf_counter()
            with contextlib.redirect_stdout(sys.stderr):
                result=engine.system_one(state=payload['state'],questions=payload['questions'])
            synchronize()
            result['latency_seconds']=time.perf_counter()-started
        except Exception as exc:
            # Do not print request text or private file/connection details into stdout.
            result={'error':type(exc).__name__}
        print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)


if __name__=='__main__':main()
