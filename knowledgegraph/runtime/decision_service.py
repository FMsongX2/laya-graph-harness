"""Local resident code -> Neo4j -> Laya -> evidence tool, independent of Cognee inference."""
import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
import time
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from decision_engine import BatchRequest, CONFIG, Engine, SelectionRequest
from graph_walk import WalkEngine, WalkGraph, WalkRequest
from lifecycle import ActivityMiddleware

ROOT = Path(__file__).resolve().parents[1]


@asynccontextmanager
async def lifespan(app):
    app.state.lock = asyncio.Lock()
    app.state.engine = await asyncio.to_thread(Engine)
    app.state.walk_engine = WalkEngine(WalkGraph(app.state.engine.graph),app.state.engine.worker)
    try:
        await asyncio.to_thread(app.state.engine.worker.start)
        yield
    finally:
        await asyncio.to_thread(app.state.engine.close)


app = FastAPI(title='KnowledgeGraph bounded local decisions', lifespan=lifespan)
app.add_middleware(ActivityMiddleware)


@app.get('/health')
async def health():
    worker = app.state.engine.worker
    ready = worker.proc is not None and worker.proc.poll() is None
    return {'service': 'knowledgegraph-decision', 'ready': ready, 'busy': app.state.lock.locked(),
            'worker_pid': worker.proc.pid if ready else None, 'worker_load_seconds': worker.load_seconds,
            'worker_device': worker.device if ready else None,
            'warmed': ready and worker.warmed_pid == worker.proc.pid, 'worker_warmup_seconds': worker.warmup_seconds,
            'warmup_model_calls': worker.warmup_calls, 'local_model_calls': worker.calls - worker.warmup_calls,
            'task': 'source_relationship', 'candidate_count': CONFIG['candidate_count'],
            'port': CONFIG['port'], 'external_model_calls': 0, 'arbitrary_multihop': False,
            'experimental_semantic_walk':True,'walk_max_hops':16,'visited_filter':'request_local_hash_set',
            'model_backend':worker.backend,'model_identity':worker.identity}


async def perform(items,walk=False):
    submitted = time.perf_counter()
    async with app.state.lock:
        start = time.perf_counter()
        # Wait for the thread to finish before releasing the worker lock, including on disconnect.
        engine=app.state.walk_engine if walk else app.state.engine
        job = asyncio.create_task(asyncio.to_thread(lambda: [engine.run(item) for item in items]))
        try:
            results = await asyncio.shield(job)
        except asyncio.CancelledError:
            await job
            raise
        except Exception as exc:
            # Do not serialize private driver/auth exception details into user-visible output.
            raise HTTPException(status_code=503, detail={'error': 'local_selection_failed',
                                                         'type': type(exc).__name__}) from None
        request_id = uuid4().hex
        measurement = {'request_id': request_id, 'items': len(items), 'queue_seconds': start - submitted,
                       'execution_seconds': time.perf_counter() - start,
                       'local_model_calls': sum(r['local_model_calls'] for r in results),
                       'routes': [r.get('route') for r in results], 'statuses': [r['status'] for r in results]}
        # Timings only: no prompts, evidence, credentials, or evaluator labels in logs.
        with (ROOT/'logs/decision-measurements.jsonl').open('a') as f:
            f.write(json.dumps(measurement) + '\n')
        return {'request_id': request_id, 'measurement': measurement, 'results': results}


@app.post('/select')
async def select_relationship(body: SelectionRequest):
    result = await perform([body])
    return {'request_id': result['request_id'], 'measurement': result['measurement'], **result['results'][0]}


@app.post('/walk')
async def walk_relationships(body: WalkRequest):
    result=await perform([body],walk=True)
    return {'request_id':result['request_id'],'measurement':result['measurement'],**result['results'][0]}


@app.post('/warmup')
async def warmup():
    async with app.state.lock:
        job = asyncio.create_task(asyncio.to_thread(app.state.engine.worker.warmup))
        try:
            await asyncio.shield(job)
        except asyncio.CancelledError:
            await job
            raise
    return await health()


@app.post('/select-batch')
async def select_batch(body: BatchRequest):
    return await perform(body.items)
