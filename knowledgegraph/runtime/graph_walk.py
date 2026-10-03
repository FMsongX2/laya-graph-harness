"""Experimental local semantic walk with request-local visited-node filtering."""
from __future__ import annotations
import math
import re
import time
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from decision_engine import CONFIG, RELATIONS, Graph, Worker, rank_eight


class WalkRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    task: Literal['semantic_walk']='semantic_walk'
    goal: str=Field(min_length=20,max_length=1000)
    start_node_id: str | None=Field(default=None,min_length=1,max_length=200)
    paper_id: str | None=Field(default=None,pattern=r'^P\d{2}$')
    target_node_id: str | None=Field(default=None,min_length=1,max_length=200)
    max_hops: int=Field(default=4,ge=1,le=16)
    include_candidates: bool=False

    @model_validator(mode='after')
    def starting_point(self):
        if bool(self.start_node_id)==bool(self.paper_id):
            raise ValueError('Provide exactly one start_node_id or paper_id')
        return self


class WalkGraph:
    """Reuse the live graph connection; never set global visited properties."""
    def __init__(self,base):self.base=base

    def node(self,node_id):
        rows=self.base.query('MATCH (n:__Node__ {id:$id}) RETURN n.id AS node_id,n.name AS name,'
                             'EXISTS { MATCH (n)-[:source_reviewed_against]->() } AS is_assertion,'
                             'EXISTS { MATCH (n)<-[:source_reviewed_against]-() } AS is_paper LIMIT 1',id=node_id)
        return rows[0] if rows else None

    def paper(self,paper_id):
        rows=self.base.query('MATCH (p:__Node__) WHERE toLower(p.name) STARTS WITH $prefix '
                             'AND EXISTS { MATCH (p)<-[:source_reviewed_against]-() } '
                             'RETURN p.id AS node_id,p.name AS name ORDER BY p.id LIMIT 2',prefix=paper_id.lower()+':')
        if len(rows)>1:raise RuntimeError('Paper scope resolves to multiple graph nodes')
        return dict(rows[0],is_paper=True,is_assertion=False) if rows else None

    def candidates(self,current,visited,limit):
        if current.get('is_paper'):
            match='MATCH (p:__Node__ {id:$id})<-[review:source_reviewed_against]-(a:__Node__)<-[:has_assertion]-(s:__Node__) MATCH (a)-[r]->(t:__Node__) '
        elif current.get('is_assertion'):
            match='MATCH (s:__Node__)-[:has_assertion]->(a:__Node__ {id:$id})-[r]->(t:__Node__) MATCH (a)-[review:source_reviewed_against]->(p:__Node__) '
        else:
            match='MATCH (s:__Node__ {id:$id})-[:has_assertion]->(a:__Node__)-[r]->(t:__Node__) MATCH (a)-[review:source_reviewed_against]->(p:__Node__) '
        # Filter in Neo4j before LIMIT, then independently filter again in Python.
        rows=self.base.query(match+'WHERE type(r) IN $relations AND NOT t.id IN $visited '
                             'AND NOT a.id IN $visited '
                             'RETURN DISTINCT p.id AS paper_node,p.name AS paper_name,a.id AS node_id,a.name AS name,'
                             's.id AS source_id,t.id AS target_id,s.name AS source,s.description AS source_display,'
                             'type(r) AS relation,t.name AS target,t.description AS target_display '
                             'ORDER BY a.id LIMIT $limit',id=current['node_id'],relations=RELATIONS,
                             visited=list(visited),limit=limit+1)
        for row in rows:
            row['assertion_id']=row['name'].split(' @',1)[0].upper()
            for key in ('source','target'):
                display=row.get(key+'_display')
                if isinstance(display,str) and display.casefold()==row[key].casefold():row[key]=display
            row['description']=f"{row['source']} | {row['relation'].replace('_',' ')} | {row['target']}"
        return rows

    def proof(self,row):return self.base.proof(row)


class WalkEngine:
    """Choose locally, read evidence, build the next state, and repeat without an agent round-trip."""
    def __init__(self,graph,worker):self.graph=graph;self.worker=worker

    def run(self,request: WalkRequest):
        started=time.perf_counter();visited=set();path=[];history=[];trace=[];model_calls=0;filtered_total=0
        result={'task':'semantic_walk','route':'local_walk','experimental':True,'answer_generated':False,
                'external_model_calls':0,'needs_review':True,'goal_verified':False,
                'visited_scope':'request-local; no graph writes','visited_filtering':'Neo4j before LIMIT and Python before candidate JSON',
                'confidence_scope':'Uncalibrated navigation with variable candidate counts; not goal or truth certification.'}
        def finish(reason):
            result.update(status='target_reached' if reason=='target_reached' else 'bounded_walk',
                          target_node_reached=reason=='target_reached',
                          stop_reason=reason,path=path,trace=trace,visited_node_ids=sorted(visited),
                          visited_tags=[{'node_id':node,'tag':'visited'} for node in sorted(visited)],
                          local_model_calls=model_calls,filtered_revisits=filtered_total,
                          hops=len(path),timing={'total_seconds':time.perf_counter()-started})
            return result
        if not request.goal.strip() or re.search(r'[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]',request.goal):
            return finish('expected_short_english_goal')
        current=self.graph.node(request.start_node_id) if request.start_node_id else self.graph.paper(request.paper_id)
        if not current:return finish('start_not_found')
        if current.get('is_assertion'):
            seeds=self.graph.candidates(current,set(),2)
            if len(seeds)!=1:return finish('ambiguous_assertion_start')
            result['seed_proof']=self.graph.proof(seeds[0]);visited.add(current['node_id'])
            history.append({'relationship':seeds[0]['description'],'proof':result['seed_proof']})
            current=self.graph.node(seeds[0]['target_id'])
            if not current:return finish('selected_target_missing')
        visited.add(current['node_id']);result['start']=current
        if request.target_node_id==current['node_id']:return finish('target_reached')
        for hop in range(request.max_hops):
            rows=self.graph.candidates(current,visited,CONFIG['paper_pool_limit'])
            if len(rows)>CONFIG['paper_pool_limit']:return finish('candidate_pool_limit_exceeded')
            pool=[];seen=set();filtered=0
            for row in rows:
                duplicate=row['node_id'] in seen
                old_assertion=row['node_id'] in visited
                if duplicate or old_assertion or row['target_id'] in visited:
                    filtered+=1;continue
                seen.add(row['node_id']);pool.append(row)
            filtered_total+=filtered
            if not pool:return finish('no_unvisited_candidates')
            menu=rank_eight(pool,request.goal)
            criteria={f'c{i}':row['description'][:180] for i,row in enumerate(menu)}
            if len(set(criteria.values()))!=len(criteria):return finish('indistinguishable_candidate_descriptions')
            recent=[{'relationship':p['relationship'],'quote':p['proof']['evidence'][0]['quote'][:240],
                     'conditions':[c['condition'][:120] for c in p['proof']['conditions'][:2]]} for p in history[-3:]]
            payload={'state':{'goal':request.goal,'current_node':{'node_id':current['node_id'],'name':current['name']},
                              'recent_evidence':recent},
                     'questions':{'action':{'type':'choice',
                        'instructions':'Choose the next reviewed relationship relevant to the goal and evidence. Preserve relationship direction and scope. The options are source entity | relationship | target entity.',
                        'criteria':criteria}}}
            response=None;model_seconds=0.0
            if len(menu)==1:key='c0';method='single_unvisited_candidate'
            else:
                response,model_seconds=self.worker.evaluate(payload);model_calls+=1;method='laya'
                usage=response.get('usage') or {}
                if usage.get('truncated') or usage.get('state_tokens_dropped',0) or usage.get('options'):
                    return finish('model_input_truncated_or_options_collapsed')
                answer=response.get('answers',{}).get('action',{});key=answer.get('choice');probs=answer.get('probabilities',{})
                if key not in criteria or set(probs)!=set(criteria):raise RuntimeError('Laya returned an unavailable walk action')
                if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1 for v in probs.values()) or abs(sum(probs.values())-1)>.02:
                    raise RuntimeError('Laya returned invalid walk probabilities')
            selected=menu[int(key[1:])]
            proof=self.graph.proof(selected)
            next_node=self.graph.node(selected['target_id'])
            if not next_node:return finish('selected_target_missing')
            before=set(visited)
            visited.update((selected['node_id'],next_node['node_id']))
            step={'hop':hop,'from_node':current['node_id'],'assertion_node':selected['node_id'],'to_node':next_node['node_id'],
                  'transition':'paper_seed' if current.get('is_paper') else 'assertion_seed' if current.get('is_assertion') else 'semantic_relation',
                  'relationship':selected['description'],'proof':proof,'method':method,'model_seconds':model_seconds}
            path.append(step)
            history.append({'relationship':selected['description'],'proof':proof})
            audit={'hop':hop,'candidate_assertion_ids':[row['node_id'] for row in menu],
                   'candidate_target_ids':[row['target_id'] for row in menu],'visited_before':sorted(before),
                   'filtered_revisits':filtered,'choice':selected['node_id'],'method':method}
            if request.include_candidates:audit['model_payload']=payload
            trace.append(audit);current=next_node
            if request.target_node_id and current['node_id']==request.target_node_id:return finish('target_reached')
        return finish('hop_limit')
