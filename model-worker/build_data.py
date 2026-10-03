"""Bounded request renderer extracted from the tested local worker."""
import json

def compact(request):
    req = json.loads(json.dumps(request))
    state = req['state']
    if isinstance(state, dict) and 'read_documents' in state:
        state['read_documents'] = [dict(id=d.get('id'), title=d.get('title'),
            text=d.get('text', '')[:600]) for d in state['read_documents']]
    for q in req['questions'].values():
        if q['type'] != 'choice':
            raise ValueError('This pilot supports choice only')
        q['criteria'] = {k: v[:180] if isinstance(v, str) else v for k,v in q['criteria'].items()}
    return req
