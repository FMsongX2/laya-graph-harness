"""Provider switches must preserve selection identity, evidence and failure boundaries."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
import decision_engine as decision
from decision_engine import Engine,SelectionRequest,Worker
from graph_walk import WalkEngine,WalkRequest
from test_decision_contract import FakeGraph,FakeWorker,EXCERPT
from test_graph_walk import FakeGraph as WalkGraph,FakeWorker as WalkWorker,GOAL


class BackendContracts(unittest.TestCase):
    def test_new_provider_is_identified_and_not_given_laya_calibration(self):
        class CertainWorker(FakeWorker):
            backend='decision2'
            def evaluate(self,payload):
                response,elapsed=super().evaluate(payload)
                answer=response['answers']['action'];answer['probabilities']={k:float(k==answer['choice']) for k in answer['probabilities']}
                return response,elapsed
        worker=CertainWorker();engine=Engine(FakeGraph(),worker)
        result=engine.run(SelectionRequest(paper_id='P01',source_excerpt=EXCERPT,policy='model'))
        self.assertEqual(result['model_backend'],'decision2');self.assertEqual(result['route'],'decision2')
        self.assertTrue(result['needs_review']);self.assertIn('Uncalibrated',result['confidence_scope'])
        self.assertEqual(result['selected']['conditions'][0]['condition'],'Only in the reported experimental scope.')

    def test_explicit_laya_policy_is_not_silently_routed_to_another_provider(self):
        worker=FakeWorker();worker.backend='decision2'
        result=Engine(FakeGraph(),worker).run(SelectionRequest(paper_id='P01',source_excerpt=EXCERPT,policy='laya'))
        self.assertEqual(worker.calls,0);self.assertIsNone(result['selected'])
        self.assertEqual(result['stop_reason'],'requested_laya_backend_unavailable')

    def test_native_question_error_returns_review_without_reading_selected_proof(self):
        class RejectWorker:
            backend='decision2'
            def evaluate(self,payload):return {'answers':{'action':{'error':'max_length_exceeded'}}},.01
        result=Engine(FakeGraph(),RejectWorker()).run(SelectionRequest(paper_id='P01',source_excerpt=EXCERPT,policy='model'))
        self.assertIsNone(result['selected']);self.assertEqual(result['stop_reason'],'model_question_rejected')
        walk=WalkEngine(WalkGraph(),RejectWorker()).run(WalkRequest(start_node_id='A',goal=GOAL))
        self.assertEqual(walk['hops'],0);self.assertEqual(walk['stop_reason'],'model_question_rejected')

    def test_walk_preserves_feedback_and_visited_rules_for_decision2(self):
        worker=WalkWorker();worker.backend='decision2'
        result=WalkEngine(WalkGraph(),worker).run(WalkRequest(start_node_id='A',goal=GOAL,max_hops=2))
        self.assertEqual([s['method'] for s in result['path']],['decision2','decision2'])
        self.assertEqual(worker.requests[1]['state']['current_node']['node_id'],'B')
        self.assertTrue(worker.requests[1]['state']['recent_evidence'])
        for trace in result['trace']:
            self.assertFalse(set(trace['visited_before']).intersection(trace['candidate_assertion_ids']+trace['candidate_target_ids']))

    def test_backend_command_requires_local_package_and_preserves_device_configuration(self):
        with tempfile.TemporaryDirectory() as temp:
            package=Path(temp)/'package';package.mkdir();(package/'MODEL_MANIFEST.json').write_text('{}')
            config={**decision.CONFIG,'backend':'decision2','worker_python':sys.executable,
                    'decision2':{'model':str(package),'device':'mps','threads':2,'mps_memory_fraction':.4}}
            with patch.dict(decision.CONFIG,config,clear=True):
                command=Worker().command()
                self.assertIn('decision2_worker.py',command[1]);self.assertEqual(command[command.index('--device')+1],'mps')
                self.assertEqual(command[command.index('--mps-memory-fraction')+1],'0.4')
                (package/'MODEL_MANIFEST.json').unlink()
                with self.assertRaises(RuntimeError):Worker().command()
            for options in [{'device':'https://remote.invalid'}, {'threads':True}, {'mps_memory_fraction':2}]:
                (package/'MODEL_MANIFEST.json').write_text('{}')
                bad={**config,'decision2':{**config['decision2'],**options}}
                with patch.dict(decision.CONFIG,bad,clear=True):
                    with self.assertRaises(ValueError):Worker().command()


if __name__=='__main__':unittest.main()
