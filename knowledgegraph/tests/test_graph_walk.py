"""Loop state, cycle filtering, bounded termination and independent requests."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from graph_walk import WalkEngine, WalkRequest

GOAL='Follow connected relationships and collect supporting evidence for the retrieval question.'

def row(ident,source,target):
    return {'node_id':ident,'source_id':source,'target_id':target,
            'description':f'source {source} | uses method | target {target}'}

class FakeGraph:
    def __init__(self):
        self.nodes={n:{'node_id':n,'name':n,'is_paper':False,'is_assertion':False} for n in 'ABCDEF'}
        self.edges={'A':[row('ab','A','B'),row('ad','A','D'),row('self','A','A'),row('ab','A','B')],
                    'B':[row('ba','B','A'),row('bc','B','C'),row('be','B','E'),row('ab','A','B')],
                    'C':[row('ca','C','A'),row('cb','C','B'),row('cd','C','D')],
                    'D':[],'E':[],'F':[]}
    def node(self,n):return copy.deepcopy(self.nodes.get(n))
    def paper(self,n):return self.node('A') if n=='P01' else None
    def candidates(self,current,visited,limit):return copy.deepcopy(self.edges.get(current['node_id'],[]))
    def proof(self,r):return {'node_id':r['node_id'],'evidence':[{'quote':'Synthetic supporting evidence for '+r['target_id']}],
                             'conditions':[{'condition':'Synthetic fixture only.'}]}

class FakeWorker:
    def __init__(self):self.requests=[];self.invalid=False;self.truncated=False
    def evaluate(self,payload):
        self.requests.append(copy.deepcopy(payload));criteria=payload['questions']['action']['criteria']
        desired={'A':'B','B':'C'}.get(payload['state']['current_node']['node_id'])
        key=next((k for k,v in criteria.items() if v.endswith('target '+str(desired))),next(iter(criteria)))
        return {'usage':{'truncated':self.truncated},'answers':{'action':{'choice':'invalid' if self.invalid else key,
                'probabilities':{k:float(k==key) for k in criteria}}}},.01

class WalkContract(unittest.TestCase):
    def setUp(self):
        self.graph=FakeGraph();self.worker=FakeWorker();self.engine=WalkEngine(self.graph,self.worker)
        self.req=WalkRequest(start_node_id='A',goal=GOAL,max_hops=8,include_candidates=True)

    def test_cycles_duplicates_and_visited_targets_are_not_in_next_menu(self):
        result=self.engine.run(self.req)
        self.assertEqual([s['to_node'] for s in result['path']],['B','C','D'])
        self.assertEqual(result['stop_reason'],'no_unvisited_candidates')
        self.assertEqual(result['local_model_calls'],2)
        for hop in result['trace']:
            visited=set(hop['visited_before'])
            self.assertFalse(visited.intersection(hop['candidate_assertion_ids']))
            self.assertFalse(visited.intersection(hop['candidate_target_ids']))
            self.assertEqual(len(hop['candidate_assertion_ids']),len(set(hop['candidate_assertion_ids'])))
        self.assertGreater(result['filtered_revisits'],0)

    def test_next_model_state_contains_previous_evidence_and_same_goal(self):
        self.engine.run(self.req)
        self.assertEqual(len(self.worker.requests),2)
        second=self.worker.requests[1]['state']
        self.assertEqual(second['current_node']['node_id'],'B');self.assertEqual(second['goal'],GOAL)
        self.assertIn('supporting evidence',second['recent_evidence'][0]['quote'])

    def test_tags_are_independent_per_request_and_do_not_modify_graph(self):
        original=copy.deepcopy(self.graph.edges)
        first=self.engine.run(self.req);second=self.engine.run(self.req)
        self.assertEqual(first['visited_node_ids'],second['visited_node_ids'])
        self.assertEqual(first['path'][0]['to_node'],second['path'][0]['to_node'])
        self.assertEqual(self.graph.edges,original)

    def test_hop_limit_and_explicit_target_stop(self):
        limited=self.engine.run(self.req.model_copy(update={'max_hops':1}))
        self.assertEqual(limited['hops'],1);self.assertEqual(limited['stop_reason'],'hop_limit')
        target=self.engine.run(self.req.model_copy(update={'target_node_id':'B'}))
        self.assertEqual(target['hops'],1);self.assertTrue(target['target_node_reached'])
        self.assertFalse(target['goal_verified'])

    def test_single_candidate_avoids_an_unnecessary_model_call(self):
        self.graph.edges['A']=[row('ab','A','B')];self.graph.edges['B']=[]
        result=self.engine.run(self.req)
        self.assertEqual(result['hops'],1);self.assertEqual(result['local_model_calls'],0)

    def test_invalid_or_truncated_model_output_cannot_advance(self):
        self.worker.invalid=True
        with self.assertRaises(RuntimeError):self.engine.run(self.req)
        self.worker.invalid=False;self.worker.truncated=True
        result=self.engine.run(self.req)
        self.assertEqual(result['hops'],0);self.assertIn('truncated',result['stop_reason'])

    def test_missing_start_and_invalid_request_are_bounded(self):
        self.assertEqual(self.engine.run(self.req.model_copy(update={'start_node_id':'absent'}))['stop_reason'],'start_not_found')
        for change in ({'paper_id':'P01'},{'start_node_id':None},{'max_hops':17},{'task':'unrestricted_agent'}):
            with self.assertRaises(ValueError):WalkRequest(**{**self.req.model_dump(),**change})

    def test_assertion_start_is_resolved_before_tagging_candidates(self):
        self.graph.nodes['ab']={'node_id':'ab','name':'ab','is_assertion':True,'is_paper':False}
        self.graph.edges['ab']=[row('ab','A','B')]
        result=self.engine.run(self.req.model_copy(update={'start_node_id':'ab'}))
        self.assertIn('ab',result['visited_node_ids']);self.assertEqual(result['start']['node_id'],'B')
        self.assertIn('seed_proof',result)
        self.assertIn('supporting evidence for B',self.worker.requests[0]['state']['recent_evidence'][0]['quote'])
        for hop in result['trace']:self.assertNotIn('ab',hop['candidate_assertion_ids'])

if __name__=='__main__':unittest.main()
