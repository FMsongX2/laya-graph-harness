#!/usr/bin/env python3
"""Local skill bootstrap. No corpus prompts, user input, or credentials in logs."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

PACKAGE_ROOT = Path(__file__).resolve().parents[5]
KG = Path(os.environ.get('LAYA_GRAPH_KG_ROOT', str(PACKAGE_ROOT/'knowledgegraph'))).expanduser().resolve()/'kg'
LOG = KG.parent/'logs/domain-alignment-bootstrap.jsonl'


def should_prepare(event):
    name = event.get('hook_event_name')
    if name == 'UserPromptSubmit':
        return bool(re.search(r'(?<![\w-])\$(?:knowledgegraph-laya:)?(?:domain-alignment|bounded-selection)(?![\w-])', str(event.get('prompt', ''))))
    if name != 'PreToolUse': return False
    tool = str(event.get('tool_name', ''))
    data = event.get('tool_input') or {}
    if isinstance(data, str):
        try: data = json.loads(data)
        except ValueError: data = {'code': data}
    if not isinstance(data, dict): return False
    if tool == 'Skill':
        return str(data.get('skill', '')).split(':')[-1] in {'domain-alignment', 'bounded-selection'}
    if tool in {'Read', 'read_file'} or tool.endswith('__read_file'):
        path = data.get('file_path') or data.get('path') or ''
        return any(str(path).replace('\\', '/').endswith('/'+name+'/SKILL.md') for name in ['domain-alignment','bounded-selection'])
    if tool not in {'Bash', 'exec_command', 'functions.exec', 'exec'}: return False
    # Observe reader tool arguments only, never parse/replay arbitrary shell code.
    command = data.get('command') or data.get('cmd') or data.get('code') or data.get('input') or ''
    if not isinstance(command, str): return False
    return (bool(re.search(r'(?:cat|sed|head|read_text|readFile)\b', command)) and
            bool(re.search(r'(?:domain-alignment|bounded-selection)[/\\]SKILL\.md', command)))


def prepare():
    t = time.perf_counter()
    proc = subprocess.run([str(KG), 'prepare'], stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=360)
    if proc.returncode: raise RuntimeError('Local runtime preparation failed; inspect kg status and decision-status')
    result = json.loads(proc.stdout)
    if not result.get('ready'): raise RuntimeError('Local runtime is not ready')
    return {'ready': True, 'embedding_loaded': result.get('embedding_loaded'),
            'graph_ready': result['graph']['ready'], 'laya_ready': result['decision']['ready'],
            'laya_warmed': result['decision']['warmed'],
            'knowledge_pid': result['knowledge_pid'], 'worker_pid': result['decision']['worker_pid'],
            'seconds': time.perf_counter() - t}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hook', action='store_true')
    args = parser.parse_args()
    event = None
    if args.hook:
        try: event = json.load(sys.stdin)
        except (ValueError, TypeError): return
        if not isinstance(event, dict) or not should_prepare(event): return
    try:
        result = prepare()
        LOG.parent.mkdir(exist_ok=True)
        with LOG.open('a') as output:
            output.write(json.dumps({'event': event.get('hook_event_name') if event else 'skill_preprocess',
                                     **result}) + '\n')
        # Stable rendered content avoids re-injecting an unchanged skill just for timing noise.
        if not args.hook: print(json.dumps({k: v for k, v in result.items() if k != 'seconds'}))
    except Exception:
        if args.hook:
            # An initialization failure must not block loading the skill or change approvals.
            print('Domain-Alignment runtime preparation failed; check kg status and decision-status.', file=sys.stderr)
        else:
            print(json.dumps({'ready': False, 'error': 'local_runtime_preparation_failed'}))


if __name__ == '__main__': main()
