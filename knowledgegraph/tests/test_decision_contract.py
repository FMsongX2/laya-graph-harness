"""Failure cases that must not silently produce an executable selection."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
from decision_engine import Engine, Graph, SelectionRequest

EXCERPT = 'The source describes how the reviewed method selects source evidence for the retrieval task.'


class FakeGraph:
    def __init__(self, count=8):
        self.rows = [{'node_id': str(i), 'description': f'Method {i} | uses metric | unique metric {i}'} for i in range(count)]
        self.links = []
    def claims(self, paper): return self.rows
    def exact_links(self, paper, excerpt): return self.links
    def proof(self, row):
        return {'node_id': row['node_id'], 'evidence': [{'quote': EXCERPT}],
                'conditions': [{'condition': 'Only in the reported experimental scope.'}],
                'evidence_truncated': False, 'conditions_truncated': False}


class FakeWorker:
    def __init__(self): self.calls = 0; self.last = None; self.bad_key = False; self.truncated = False
    def evaluate(self, req):
        self.calls += 1; self.last = copy.deepcopy(req)
        keys = list(req['questions']['action']['criteria'])
        return {'usage': {'truncated': self.truncated}, 'answers': {'action': {
            'choice': 'c999' if self.bad_key else keys[0],
            'probabilities': {k: 1 / len(keys) for k in keys}}}}, .01


class Contract(unittest.TestCase):
    def setUp(self):
        self.graph, self.worker = FakeGraph(), FakeWorker()
        self.engine = Engine(graph=self.graph, worker=self.worker)
        self.req = SelectionRequest(paper_id='P03', source_excerpt=EXCERPT, policy='laya')

    def test_choice_outside_menu_is_error(self):
        self.worker.bad_key = True
        with self.assertRaises(RuntimeError): self.engine.run(self.req)

    def test_truncation_returns_no_selected_node(self):
        self.worker.truncated = True
        r = self.engine.run(self.req)
        self.assertIsNone(r['selected']); self.assertTrue(r['needs_review'])

    def test_fewer_than_eight_does_not_call_model(self):
        self.graph.rows.pop()
        r = self.engine.run(self.req)
        self.assertIsNone(r['selected']); self.assertEqual(self.worker.calls, 0)

    def test_indistinguishable_description_rejected(self):
        self.graph.rows[1]['description'] = self.graph.rows[0]['description']
        r = self.engine.run(self.req)
        self.assertIsNone(r['selected']); self.assertEqual(self.worker.calls, 0)

    def test_low_confidence_preserves_full_conditions_and_alternatives(self):
        r = self.engine.run(self.req)
        self.assertTrue(r['needs_review']); self.assertEqual(len(r['candidates']), 8)
        self.assertEqual(r['selected']['conditions'][0]['condition'], 'Only in the reported experimental scope.')
        self.assertEqual(set(self.worker.last), {'state', 'questions'})

    def test_exact_link_set_is_not_forced_to_single_choice(self):
        self.graph.links = [{'node_id': '0'}, {'node_id': '1'}]
        r = self.engine.run(self.req.model_copy(update={'policy': 'auto'}))
        self.assertEqual(r['status'], 'linked_set'); self.assertIsNone(r['selected'])
        self.assertEqual(len(r['matches']), 2); self.assertEqual(self.worker.calls, 0)

    def test_unknown_fields_and_unbounded_input_rejected(self):
        for extra in [{'goal_node': 'hidden_gold'}, {'source_excerpt': 'a' * 1001}, {'task': 'arbitrary_navigation'}]:
            with self.assertRaises(ValueError): SelectionRequest(**{**self.req.model_dump(), **extra})

    def test_approved_coordinates_preserved_and_stale_quote_rejected(self):
        graph = Graph.__new__(Graph)
        graph.query = lambda query, **params: ([{'evidence_name': 'p03-a001 pdf p4 evidence 0', 'quote': EXCERPT}]
                                               if 'supported_by' in query else [{'condition': 'Reported scope only.'}])
        node = {'assertion_id': 'P03-A001', 'node_id': 'actual-node', 'name': 'p03-a001 @aaaaaaaaaaaa',
                'source': 'Method', 'target': 'Model', 'relation': 'uses_model', 'paper_name': 'P03: Paper'}
        approved = {'id': 'P03-A001', 'source': {'name': 'Method'}, 'target': {'name': 'Model'},
                    'relation': 'uses_model', 'verification_status': 'source_reviewed', 'pdf_sha256': 'a'*64,
                    'paper_id': 'P03', 'paper_title': 'Paper', 'original_pdf': '/test/paper.pdf',
                    'conditions': ['Reported scope only.'], 'evidence': [{'page': 4, 'quote': EXCERPT, 'row_label': 'actual row'}]}
        with patch.object(Path, 'read_text', return_value=json.dumps(approved)):
            result = graph.proof(node)
            self.assertEqual(result['evidence'][0]['source_coordinates'][0]['page'], 4)
            self.assertEqual(result['evidence'][0]['source_coordinates'][0]['row_label'], 'actual row')
        approved['evidence'][0]['quote'] = 'Changed approved text.'
        with patch.object(Path, 'read_text', return_value=json.dumps(approved)):
            with self.assertRaises(RuntimeError): graph.proof(node)


if __name__ == '__main__': unittest.main()
