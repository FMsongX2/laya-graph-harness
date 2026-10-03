"""Bounded local source-excerpt selection. No evaluator data or generated answers."""
from __future__ import annotations

import base64
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import select
import sqlite3
import subprocess
import time
import urllib.request

from cryptography.fernet import Fernet
from neo4j import GraphDatabase, Query
from neo4j.exceptions import ServiceUnavailable, SessionExpired
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / 'config/decision.json').read_text())
SETTINGS = json.loads((ROOT / 'config/settings.json').read_text())
INSTRUCTIONS = ('Select a reviewed relationship supported by the source excerpt. '
                'Match the meaning, direction, and scope. '
                'The option format is source entity | relationship | target entity.')
RELATIONS = ['solves', 'requires', 'uses_preprocessing', 'uses_model', 'implemented_by',
             'alternative_to', 'evaluated_on', 'has_limitation', 'reports_result', 'works_by',
             'supports', 'has_tradeoff', 'uses_parameter', 'uses_metric', 'depends_on',
             'has_property', 'uses_dataset', 'uses_method', 'has_scope', 'mitigates']


class SelectionRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    task: Literal['source_relationship'] = 'source_relationship'
    paper_id: str = Field(pattern=r'^P\d{2}$')
    source_excerpt: str = Field(min_length=60, max_length=1000)
    policy: Literal['auto', 'laya'] = 'auto'
    include_candidates: bool = False


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    items: list[SelectionRequest] = Field(min_length=1, max_length=CONFIG['max_batch_items'])


class Worker:
    """One persistent, serialized JSONL worker; discard the pipe on timeout/error."""
    def __init__(self):
        self.proc = None
        self.log = None
        self.load_seconds = None
        self.calls = 0
        self.warmup_calls = 0
        self.warmed_pid = None
        self.warmup_seconds = None

    def start(self):
        if self.proc and self.proc.poll() is None:
            return
        self.close()
        lab = (ROOT / CONFIG['model_lab']).resolve()
        # Preserve the venv executable symlink: resolving it would lose its site-packages.
        configured_python = ROOT / CONFIG['worker_python']
        python = configured_python.parent.resolve() / configured_python.name
        adapter = lab / CONFIG['adapter']
        if not all(p.exists() for p in [python, lab / 'worker.py', lab / 'base-model', adapter / 'head.safetensors']):
            raise RuntimeError('Configured local Laya runtime/checkpoint is missing')
        env = {k: os.environ[k] for k in ['PATH', 'HOME', 'USER', 'LANG', 'TMPDIR'] if k in os.environ}
        env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
        (ROOT / 'logs').mkdir(exist_ok=True)
        self.log = (ROOT / 'logs/decision-worker.log').open('a')
        t = time.perf_counter()
        self.proc = subprocess.Popen([str(python), str(lab / 'worker.py'), '--model', str(lab / 'base-model'),
                                      '--head', str(adapter)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.log, text=True, bufsize=1, env=env)
        try:
            ready = self.read()
            if ready.get('ready') is not True:
                raise RuntimeError('Local Laya worker did not become ready')
        except Exception:
            self.close(); raise
        self.load_seconds = time.perf_counter() - t

    def read(self):
        if not select.select([self.proc.stdout], [], [], CONFIG['worker_timeout_seconds'])[0]:
            raise RuntimeError('Local Laya worker timed out')
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError('Local Laya worker exited')
        try:
            value = json.loads(line)
        except ValueError as exc:
            raise RuntimeError('Invalid JSON from local Laya worker') from exc
        if not isinstance(value, dict) or 'error' in value:
            raise RuntimeError('Local Laya worker rejected the request')
        return value

    def evaluate(self, request):
        self.start()
        t = time.perf_counter()
        try:
            self.proc.stdin.write(json.dumps(request, ensure_ascii=False) + '\n')
            self.proc.stdin.flush()
            response = self.read()
        except Exception:
            self.close(); raise
        self.calls += 1
        return response, time.perf_counter() - t

    def warmup(self):
        """Prime local inference once per worker, without reading corpus/evaluator data."""
        self.start()
        if self.warmed_pid == self.proc.pid:
            return
        payload = {'state': {'goal': 'Which reviewed relationship is grounded in this source excerpt?\n'
                   'The retrieval method reads document evidence from an index and checks the relationship '
                   'under the experimental conditions described by the source.'},
                   'questions': {'action': {'type': 'choice', 'instructions': INSTRUCTIONS,
                       'criteria': {f'c{i}': f'Method {i} | uses method | Evidence retrieval variant {i}' for i in range(8)}}}}
        _, elapsed = self.evaluate(payload)
        self.warmup_calls += 1
        self.warmed_pid = self.proc.pid
        self.warmup_seconds = elapsed

    def close(self):
        if self.proc:
            if self.proc.poll() is None:
                self.proc.stdin.close()
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.terminate()
                    try: self.proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.proc.kill(); self.proc.wait(timeout=5)
            self.proc.stdout.close()
            self.proc = None
        if self.log:
            self.log.close(); self.log = None


class Graph:
    """Fixed read queries against the configured Cognee semantic dataset only."""
    def __init__(self):
        dataset = SETTINGS['knowledge']['semantic_dataset']
        with sqlite3.connect(f'file:{ROOT}/state/system/databases/cognee_db?mode=ro', uri=True) as c:
            row = c.execute('SELECT b.graph_database_url,b.graph_database_name,b.graph_database_connection_info '
                            'FROM dataset_database b JOIN datasets d ON d.id=b.dataset_id '
                            "WHERE d.name=? AND b.graph_database_provider='neo4j'", (dataset,)).fetchone()
        if row is None:
            raise RuntimeError('The configured semantic dataset is not backed by Neo4j')
        key = json.loads((ROOT / SETTINGS['graph']['encryption_key_file']).read_text())['encryption_key']
        info = json.loads(row[2])
        cipher = Fernet(base64.urlsafe_b64encode(hashlib.sha256(key.encode()).digest()))
        password = cipher.decrypt(info['graph_database_password'].encode()).decode()
        self.driver = GraphDatabase.driver(row[0], auth=(info['graph_database_username'], password),
                                           connection_timeout=3, max_connection_pool_size=2)
        self.database = row[1]

    def query(self, query, **params):
        for attempt in range(2):
            try:
                with self.driver.session(database=self.database, default_access_mode='READ') as session:
                    return session.run(Query(query, timeout=10), **params).data()
            except (ServiceUnavailable, SessionExpired):
                if attempt:
                    raise RuntimeError('Semantic Neo4j graph is unavailable') from None
                # Let Cognee manage its own dataset container and existing two-container limit.
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                try:
                    with opener.open(f"http://127.0.0.1:{SETTINGS['server']['port']}/kg/graph-ready", timeout=190) as r:
                        json.load(r)
                except Exception:
                    raise RuntimeError('Semantic graph is unavailable; run ./kg up and retry') from None

    def claims(self, paper_id):
        rows = self.query('MATCH (p:__Node__)<-[link:source_reviewed_against]-(a:__Node__)<-[:has_assertion]-(s:__Node__) '
                          'MATCH (a)-[r]->(t:__Node__) WHERE toLower(p.name) STARTS WITH $prefix AND type(r) IN $relations '
                          'RETURN p.id AS paper_node,p.name AS paper_name,a.id AS node_id,a.name AS name,'
                          's.name AS source,s.description AS source_display,type(r) AS relation,'
                          't.name AS target,t.description AS target_display,elementId(link) AS link_id LIMIT $limit',
                          prefix=paper_id.lower() + ':', relations=RELATIONS, limit=CONFIG['paper_pool_limit'] + 1)
        for row in rows:
            row['assertion_id'] = row['name'].split(' @', 1)[0].upper()
            for key in ['source', 'target']:
                display = row.get(key + '_display')
                if isinstance(display, str) and display.casefold() == row[key].casefold(): row[key] = display
            row['description'] = f"{row['source']} | {row['relation'].replace('_', ' ')} | {row['target']}"
        return rows

    def exact_links(self, paper_id, excerpt):
        return self.query('MATCH (a:__Node__)-[:source_reviewed_against]->(p:__Node__), '
                          '(a)-[:supported_by]->(e:__Node__) '
                          'WHERE toLower(p.name) STARTS WITH $prefix AND e.description=$text '
                          'RETURN DISTINCT a.id AS node_id', prefix=paper_id.lower() + ':', text=excerpt)

    def proof(self, node):
        evidence = self.query('MATCH (a:__Node__ {id:$id})-[:supported_by]->(e:__Node__)-[r:sourced_from]->(p:__Node__) '
                             'RETURN DISTINCT e.id AS evidence_id,e.name AS evidence_name,e.description AS quote,'
                             'p.name AS paper,p.description AS source_url,r.description AS page_reference '
                             'ORDER BY evidence_name LIMIT $limit', id=node['node_id'], limit=CONFIG['max_evidence'] + 1)
        conditions = self.query('MATCH (a:__Node__ {id:$id})-[:under_condition]->(c:__Node__) '
                               'RETURN DISTINCT c.name AS name,c.description AS condition ORDER BY name LIMIT $limit',
                               id=node['node_id'], limit=CONFIG['max_conditions'] + 1)
        if not evidence:
            raise RuntimeError('Selected graph relationship has no linked source evidence')
        # Recover reviewed page/table coordinates from the original approved record,
        # rather than guess them from a model or a missing Neo4j edge property.
        identity = node['assertion_id']
        if not re.fullmatch(r'P\d{2}-A\d+', identity):
            raise RuntimeError('Graph assertion identity is outside the approved record format')
        record_path = ROOT / 'data/semantic-assertions' / identity.split('-')[0] / (identity + '.json')
        approved = json.loads(record_path.read_text())
        if (approved['id'] != identity or approved.get('verification_status') != 'source_reviewed' or
                approved['source']['name'].casefold() != node['source'].casefold() or
                approved['target']['name'].casefold() != node['target'].casefold() or
                approved['relation'] != node['relation'] or
                not node['name'].casefold().endswith(' @' + approved['pdf_sha256'][:12].casefold())):
            raise RuntimeError('Graph relationship differs from the approved source record')
        for item in evidence:
            page = re.search(r'\bpdf p(\d+)\b', item['evidence_name'], re.IGNORECASE)
            matches = [e for e in approved['evidence'] if e['quote'] == item['quote'] and
                       (not page or e['page'] == int(page.group(1)))]
            if not matches:
                raise RuntimeError('Graph quotation differs from approved source evidence')
            item['source_coordinates'] = [{k: e[k] for k in ['page', 'evidence_type', 'table_id', 'row_label',
                                           'column_label', 'value', 'unit', 'visually_checked',
                                           'exact_quote_match', 'quote_sha256'] if k in e} for e in matches]
        if len(conditions) <= CONFIG['max_conditions'] and {c['condition'] for c in conditions} != set(approved['conditions']):
            raise RuntimeError('Graph conditions differ from the approved source record')
        return {'node_id': node['node_id'], 'assertion_id': node['assertion_id'],
                'source': node['source'], 'relation': node['relation'], 'target': node['target'],
                'paper': node['paper_name'], 'paper_id': approved['paper_id'], 'paper_title': approved['paper_title'],
                'source_review_link': True, 'verification_status': approved['verification_status'],
                'scientific_truth_certified': False, 'pdf_sha256': approved['pdf_sha256'],
                'original_pdf': approved['original_pdf'], 'approved_source_record': str(record_path),
                'evidence': evidence[:CONFIG['max_evidence']], 'conditions': conditions[:CONFIG['max_conditions']],
                'evidence_truncated': len(evidence) > CONFIG['max_evidence'],
                'conditions_truncated': len(conditions) > CONFIG['max_conditions']}

    def close(self): self.driver.close()


def rank_eight(pool, excerpt):
    """Same iterative BM25-style ranker as the measured live pilot, without gold."""
    remaining = {row['node_id']: row for row in pool}
    query = re.findall(r'\w+', excerpt.casefold())
    chosen = []
    while remaining and len(chosen) < CONFIG['candidate_count']:
        docs = {k: re.findall(r'\w+', row['description'].casefold()) for k, row in remaining.items()}
        df = Counter(w for d in docs.values() for w in set(d))
        avg = sum(map(len, docs.values())) / len(docs)
        scores = {}
        for key, tokens in docs.items():
            tf = Counter(tokens)
            scores[key] = sum(math.log(1 + (len(docs) - df[t] + .5) / (df[t] + .5)) *
                              tf[t] * 2.5 / (tf[t] + 1.5 * (.25 + .75 * len(tokens) / max(avg, 1)))
                              for t in query if tf[t])
        key = max(scores, key=scores.get)
        chosen.append(remaining.pop(key))
    return chosen


class Engine:
    def __init__(self, graph=None, worker=None):
        self.graph = graph if graph is not None else Graph()
        self.worker = worker if worker is not None else Worker()

    def run(self, request: SelectionRequest):
        t = time.perf_counter()
        result = {'task': request.task, 'paper_id': request.paper_id, 'answer_generated': False,
                  'external_model_calls': 0, 'local_model_calls': 0, 'selected': None,
                  'needs_review': True, 'steps': [], 'timing': {}}

        def finish(status, reason):
            result.update(status=status, stop_reason=reason)
            result['timing']['total_seconds'] = time.perf_counter() - t
            return result

        if not request.source_excerpt.strip() or re.search(r'[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]', request.source_excerpt):
            return finish('needs_review', 'expected_short_english_source_excerpt')
        start = time.perf_counter()
        pool = self.graph.claims(request.paper_id)
        result['timing']['graph_candidates_seconds'] = time.perf_counter() - start
        result['steps'].append('read_paper_relationships')
        if not pool:
            return finish('not_found', 'no_relationships_in_paper_scope')
        if len(pool) > CONFIG['paper_pool_limit']:
            return finish('needs_review', 'paper_pool_limit_exceeded')
        byid = {n['node_id']: n for n in pool}
        if request.policy == 'auto':
            start = time.perf_counter()
            links = self.graph.exact_links(request.paper_id, request.source_excerpt)
            result['timing']['exact_lookup_seconds'] = time.perf_counter() - start
            result['steps'].append('check_exact_source_links')
            if links:
                if any(n['node_id'] not in byid for n in links):
                    return finish('needs_review', 'source_links_outside_supported_relationship_schema')
                if len(links) > 32:
                    return finish('needs_review', 'exact_match_result_limit_exceeded')
                start = time.perf_counter()
                result['matches'] = [self.graph.proof(byid[n['node_id']]) for n in links]
                result['timing']['evidence_seconds'] = time.perf_counter() - start
                result['route'] = 'exact_source_links'
                if len(links) == 1:
                    result['selected'] = result['matches'][0]
                    result['needs_review'] = result['selected']['evidence_truncated'] or result['selected']['conditions_truncated']
                result['steps'].append('read_evidence_and_conditions')
                return finish('linked' if len(links) == 1 else 'linked_set', 'exact_source_links')
        start = time.perf_counter()
        menu = rank_eight(pool, request.source_excerpt)
        result['timing']['rank_seconds'] = time.perf_counter() - start
        result['steps'].append('rank_bounded_candidates')
        result['candidate_count'] = len(menu)
        if len(menu) != CONFIG['candidate_count']:
            return finish('needs_review', 'candidate_count_outside_calibrated_eight')
        criteria = {f'c{i}': n['description'][:180] for i, n in enumerate(menu)}
        if len(set(criteria.values())) != len(criteria):
            return finish('needs_review', 'indistinguishable_candidate_descriptions')
        payload = {'state': {'goal': 'Which reviewed relationship is grounded in this source excerpt?\n' + request.source_excerpt},
                   'questions': {'action': {'type': 'choice', 'instructions': INSTRUCTIONS, 'criteria': criteria}}}
        response, elapsed = self.worker.evaluate(payload)
        result['timing']['model_seconds'] = elapsed
        result['local_model_calls'] = 1
        result['route'] = 'laya'
        result['steps'].append('laya_choice')
        usage = response.get('usage') or {}
        result['model_usage'] = usage
        if usage.get('truncated') or usage.get('state_tokens_dropped', 0) or usage.get('options'):
            return finish('needs_review', 'model_input_truncated_or_options_collapsed')
        answer = response.get('answers', {}).get('action', {})
        key, probabilities = answer.get('choice'), answer.get('probabilities', {})
        if key not in criteria or set(probabilities) != set(criteria):
            raise RuntimeError('Laya returned a choice outside the supplied candidates')
        if not all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in probabilities.values()):
            raise RuntimeError('Laya returned invalid probabilities')
        if abs(sum(probabilities.values()) - 1) > .02:
            raise RuntimeError('Laya returned unnormalized probabilities')
        result['confidence'] = probabilities[key]
        result['confidence_scope'] = 'Eight-candidate source-relationship calibration; not a truth probability.'
        start = time.perf_counter()
        selected_node = menu[int(key[1:])]
        result['selected'] = self.graph.proof(selected_node)
        result['timing']['evidence_seconds'] = time.perf_counter() - start
        result['steps'].append('read_evidence_and_conditions')
        result['needs_review'] = (probabilities[key] < CONFIG['review_threshold'] or
                                  len(selected_node['description']) > 180 or
                                  result['selected']['evidence_truncated'] or result['selected']['conditions_truncated'])
        if request.include_candidates or result['needs_review']:
            result['candidates'] = [{'node_id': n['node_id'], 'relationship': n['description'],
                                     'model_description': criteria[f'c{i}'], 'probability': probabilities[f'c{i}'],
                                     'description_shortened': len(n['description']) > 180} for i, n in enumerate(menu)]
        return finish('candidate', 'bounded_selection_complete')

    def close(self):
        self.worker.close(); self.graph.close()
