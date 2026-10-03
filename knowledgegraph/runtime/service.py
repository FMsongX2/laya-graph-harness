"""Cognee의 로컬 추출·임베딩과 근거 조회를 제공하는 실행 서비스."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib.metadata
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
CONFIG = json.loads((PROJECT_ROOT / "config/settings.json").read_text())
SNAPSHOT = Path(CONFIG["embedding"]["snapshot"])
CHAT_MODEL = CONFIG["chat"]["model"]
EMBED_MODEL = CONFIG["embedding"]["model"]
OLLAMA = CONFIG["chat"]["upstream"]
MAX_OUTPUT_TOKENS = CONFIG["chat"]["max_output_tokens"]
EMBED_MAX_TOKENS = CONFIG["embedding"]["max_tokens"]
EMBED_DIM = CONFIG["embedding"]["dimensions"]
CPU_THREADS = CONFIG["embedding"]["cpu_threads"]

# These settings apply to this process only. The original HF snapshot is read-only input.
for key, value in {
    'HF_HOME': str(ROOT / 'cache/hf'),
    'HF_HUB_CACHE': str(ROOT / 'cache/hf/hub'),
    'HF_HUB_OFFLINE': '1',
    'HF_HUB_DISABLE_TELEMETRY': '1',
    'TRANSFORMERS_OFFLINE': '1',
    'SENTENCE_TRANSFORMERS_HOME': str(ROOT / 'cache/sentence-transformers'),
    'TORCH_HOME': str(ROOT / 'cache/torch'),
    'XDG_CACHE_HOME': str(ROOT / 'cache/xdg'),
    'MPLCONFIGDIR': str(ROOT / 'cache/matplotlib'),
    'TMPDIR': str(ROOT / 'tmp'),
    'TOKENIZERS_PARALLELISM': 'false',
}.items():
    os.environ[key] = value
for name in ('logs/requests', 'cache/hf', 'cache/torch', 'cache/xdg', 'tmp'):
    (ROOT / name).mkdir(parents=True, exist_ok=True)

import httpx
import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from lifecycle import ActivityMiddleware, tracked_operation
from sentence_transformers import SentenceTransformer


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_manifest():
    required = ['config.json', 'pytorch_model.bin', 'tokenizer.json',
                'tokenizer_config.json', 'sentencepiece.bpe.model', 'special_tokens_map.json',
                'modules.json', '1_Pooling/config.json', 'sentence_bert_config.json']
    missing = [name for name in required if not (SNAPSHOT / name).is_file()]
    if missing:
        raise RuntimeError(f'Cached BGE-M3 dense snapshot is incomplete: {missing}')
    files = []
    for path in sorted(SNAPSHOT.rglob('*')):
        if path.is_symlink() and not path.exists():
            raise RuntimeError(f'Broken cached snapshot link: {path}')
        if path.is_file():
            files.append({'name': str(path.relative_to(SNAPSHOT)), 'resolved_path': str(path.resolve()),
                          'bytes': path.stat().st_size, 'sha256': sha256_file(path)})
    return {'snapshot': str(SNAPSHOT), 'revision': SNAPSHOT.name, 'files': files,
            'dense_required_files_present': True,
            'entire_upstream_repository_snapshot_claimed_complete': False,
            'scope': 'Only dense XLM-R encoder + CLS pooling + normalization is used; sparse/ColBERT are not loaded.'}


@asynccontextmanager
async def lifespan(app):
    app.state.ready = False
    app.state.chat_lock = asyncio.Lock()
    app.state.embedding_lock = asyncio.Lock()
    app.state.knowledge_lock = asyncio.Lock()
    app.state.knowledge_operation = None
    # The local Ollama endpoint was verified to enforce JSON schemas. Advertise
    # that capability to Cognee's native adapter instead of prompted-JSON fallback.
    import litellm
    local_model = {
        'max_input_tokens': 16384, 'max_output_tokens': MAX_OUTPUT_TOKENS,
        'max_tokens': MAX_OUTPUT_TOKENS, 'input_cost_per_token': 0,
        'output_cost_per_token': 0, 'litellm_provider': 'openai', 'mode': 'chat',
        'supports_response_schema': True,
    }
    litellm.register_model({CHAT_MODEL: local_model, 'openai/' + CHAT_MODEL: local_model})
    if not litellm.supports_response_schema(model='openai/' + CHAT_MODEL):
        raise RuntimeError('Cognee did not recognize the local JSON-schema capability')
    from cognee.modules.engine.operations.setup import setup
    await setup()
    app.state.client = httpx.AsyncClient(base_url=OLLAMA, trust_env=False,
                                       follow_redirects=False, timeout=httpx.Timeout(1800, connect=5))
    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(1)
    started = perf_counter()
    manifest = snapshot_manifest()
    model = SentenceTransformer(str(SNAPSHOT), device='cpu', local_files_only=True,
                                trust_remote_code=False, cache_folder=str(ROOT / 'cache/hf'))
    model.eval()
    if model.get_sentence_embedding_dimension() != EMBED_DIM or model.max_seq_length != EMBED_MAX_TOKENS:
        raise RuntimeError('Unexpected BGE-M3 dimension or tokenization limit')
    app.state.embedder = model
    info = {'created_at': now(), 'embedding': {**manifest, 'model_name': EMBED_MODEL,
            'dimension': EMBED_DIM, 'device': str(model.device), 'dtype': str(next(model.parameters()).dtype),
            'normalize_embeddings': True, 'pooling': 'CLS token', 'max_tokens_including_special_tokens': EMBED_MAX_TOKENS,
            'over_limit_policy': 'HTTP 400; never silently truncate', 'batch_size': 4,
            'torch_num_threads': CPU_THREADS, 'torch_num_interop_threads': 1,
            'tokenizer_class': type(model.tokenizer).__name__, 'tokenizer_model_max_length': model.tokenizer.model_max_length,
            'load_seconds': perf_counter() - started},
            'chat': {'upstream': OLLAMA + '/v1/chat/completions', 'model': CHAT_MODEL,
                     'temperature': 0, 'max_tokens_cap': MAX_OUTPUT_TOKENS, 'max_concurrent_requests': 1,
                     'stream': False, 'prompt_and_response_format': 'forwarded unchanged',
                     'structured_output': 'Ollama JSON schema; locally verified',
                     'context_limit': 16384, 'context_limit_source': 'existing Ollama model parameters; not modified',
                     'input_truncation_detection': 'Not available from Ollama OpenAI response; usage and finish_reason retained'},
            'packages': {name: importlib.metadata.version(name) for name in
                         ('fastapi', 'uvicorn', 'httpx', 'sentence-transformers', 'transformers', 'torch', 'numpy', 'tokenizers')},
            'external_inference': False, 'api_key': 'local-knowledge-graph is a fake compatibility placeholder'}
    write_json(ROOT / 'model-manifest.json', info)
    app.state.manifest = info
    app.state.ready = True
    try:
        yield
    finally:
        app.state.ready = False
        await app.state.client.aclose()


app = FastAPI(title='KnowledgeGraph / Cognee', lifespan=lifespan)
app.add_middleware(ActivityMiddleware)
from knowledge_api import router as knowledge_router
app.include_router(knowledge_router)


def base_record(request, kind):
    return {'request_id': uuid4().hex, 'created_at': now(), 'kind': kind,
            'run': request.headers.get('x-run-id'),
            'engine': 'cognee',
            'stage': request.headers.get('x-stage'), 'queue_seconds': 0.0, 'execution_seconds': 0.0,
            'model': None, 'status': None}


def finish_record(record, start, submitted, effective, response):
    record['finished_at'] = now()
    record['total_seconds'] = perf_counter() - start
    path = ROOT / 'logs/requests' / (record['request_id'] + '.json')
    content = {'measurement': record, 'submitted_request': submitted,
               'effective_request': effective, 'response': response}
    if CONFIG["logging"]["request_payloads"]:
        write_json(path, content)
        record['payload_path'] = str(path)
        record['payload_sha256'] = sha256_file(path)
    else:
        record['request_sha256'] = hashlib.sha256(json.dumps(submitted, sort_keys=True).encode()).hexdigest()
    with (ROOT / 'logs/measurements.jsonl').open('a') as output:
        output.write(json.dumps(record, ensure_ascii=False) + '\n')


def error_response(record, started, submitted, effective, message, status=400):
    payload = {'error': {'message': message, 'type': 'invalid_request_error' if status < 500 else 'server_error',
                         'code': 'local_runtime_error'}}
    record.update(status=status, error=message)
    finish_record(record, started, submitted, effective, payload)
    return JSONResponse(payload, status_code=status, headers={'X-Request-ID': record['request_id']})


async def parse_body(request):
    body = await request.json()
    if not isinstance(body, dict):
        raise ValueError('Request body must be a JSON object')
    return body


@app.get('/health')
@app.get('/health/ready')
async def health():
    ready = getattr(app.state, 'ready', False)
    present = False
    reason = None
    try:
        result = await app.state.client.get('/api/tags', timeout=5)
        present = result.is_success and any(x.get('name') == CHAT_MODEL for x in result.json().get('models', []))
    except Exception as error:
        reason = str(error)
    return JSONResponse({'ready': ready and present, 'embedding_loaded': ready,
                         'ollama_model_available': present, 'chat_model': CHAT_MODEL,
                         'embedding_model': EMBED_MODEL, 'embedding_dimension': EMBED_DIM,
                         'graph_backend': CONFIG.get('graph', {}).get('provider', 'ladybug'),
                         'graph_dataset_handler': CONFIG.get('graph', {}).get('dataset_handler', 'ladybug'),
                         'chat_busy': app.state.chat_lock.locked() if ready else False,
                         'embedding_busy': app.state.embedding_lock.locked() if ready else False,
                         'error': reason, 'knowledge_operation': getattr(app.state, 'knowledge_operation', None)}, status_code=200 if ready and present else 503)


@app.get('/runtime/activity')
async def activity():
    """Idle cleanup observes local work without depending on the shared Ollama daemon."""
    return {'ready': getattr(app.state, 'ready', False),
            'chat_busy': app.state.chat_lock.locked(),
            'embedding_busy': app.state.embedding_lock.locked(),
            'knowledge_operation': app.state.knowledge_operation}


@app.get('/v1/models')
async def models():
    return {'object': 'list', 'data': [{'id': name, 'object': 'model', 'created': 0,
                                       'owned_by': 'local-knowledge-graph'} for name in (CHAT_MODEL, EMBED_MODEL)]}


@app.post('/v1/chat/completions')
async def chat(request: Request):
    started = perf_counter()
    record = base_record(request, 'chat')
    submitted = effective = None
    try:
        submitted = await parse_body(request)
        record['model'] = submitted.get('model')
        if submitted.get('model') != CHAT_MODEL:
            raise ValueError(f'Only model={CHAT_MODEL!r} is allowed; there is no fallback')
        if submitted.get('stream', False) is not False:
            raise ValueError('Streaming is unsupported; configure the framework with stream=false')
        if not isinstance(submitted.get('messages'), list) or not submitted['messages']:
            raise ValueError('messages must be a nonempty list; no prompt rewriting is performed')
        if submitted.get('n', 1) != 1:
            raise ValueError('Only n=1 is supported')
        budgets = [submitted[key] for key in ('max_tokens', 'max_completion_tokens') if submitted.get(key) is not None]
        if any(type(value) is not int or value < 1 for value in budgets):
            raise ValueError('max_tokens/max_completion_tokens must be positive integers')
        effective = dict(submitted)
        effective.pop('max_completion_tokens', None)
        effective.update(model=CHAT_MODEL, temperature=CONFIG['chat']['temperature'], stream=False,
                         max_tokens=min([MAX_OUTPUT_TOKENS] + budgets))
        record.update(temperature=CONFIG['chat']['temperature'], requested_temperature=submitted.get('temperature'),
                      max_tokens=effective['max_tokens'], requested_max_tokens=submitted.get('max_tokens'),
                      requested_max_completion_tokens=submitted.get('max_completion_tokens'),
                      input_truncation_checked=False, response_format=submitted.get('response_format'))
    except (ValueError, TypeError) as error:
        return error_response(record, started, submitted, effective, str(error))

    queued = perf_counter()
    async with app.state.chat_lock:
        acquired = perf_counter()
        record['queue_seconds'] = acquired - queued
        record['execution_started_at'] = now()
        try:
            result = await app.state.client.post('/v1/chat/completions', json=effective)
            record['execution_seconds'] = perf_counter() - acquired
            record['execution_finished_at'] = now()
            record['status'] = result.status_code
            try:
                response = result.json()
            except ValueError:
                response = {'upstream_raw_text': result.text}
            record['usage'] = response.get('usage')
            record['finish_reasons'] = [choice.get('finish_reason') for choice in response.get('choices', [])]
            record['output_truncated'] = 'length' in record['finish_reasons']
            record['upstream_model'] = response.get('model')
            finish_record(record, started, submitted, effective, response)
            return Response(content=result.content, status_code=result.status_code,
                            media_type=result.headers.get('content-type', 'application/json'),
                            headers={'X-Request-ID': record['request_id']})
        except httpx.HTTPError as error:
            record['execution_seconds'] = perf_counter() - acquired
            record['execution_finished_at'] = now()
            return error_response(record, started, submitted, effective, str(error), 502)


@tracked_operation
def embedding_work(texts):
    model = app.state.embedder
    token_lengths = [len(ids) for ids in model.tokenizer(texts, add_special_tokens=True,
                                                       truncation=False, padding=False)['input_ids']]
    if any(count > EMBED_MAX_TOKENS for count in token_lengths):
        raise ValueError(f'Embedding input exceeds {EMBED_MAX_TOKENS} tokens including special tokens: {token_lengths}')
    vectors = model.encode(texts, batch_size=4, normalize_embeddings=True, convert_to_numpy=True,
                           precision='float32', show_progress_bar=False, device='cpu')
    if vectors.shape != (len(texts), EMBED_DIM):
        raise RuntimeError(f'Unexpected embedding shape: {vectors.shape}')
    norms = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(vectors).all() or not np.allclose(norms, 1.0, atol=1e-5):
        raise RuntimeError('Nonfinite or unnormalized embeddings')
    return vectors, token_lengths, norms


@app.post('/v1/embeddings')
async def embeddings(request: Request):
    started = perf_counter()
    record = base_record(request, 'embedding')
    submitted = None
    try:
        submitted = await parse_body(request)
        record['model'] = submitted.get('model')
        if submitted.get('model') != EMBED_MODEL:
            raise ValueError(f'Only model={EMBED_MODEL!r} is allowed; there is no fallback')
        value = submitted.get('input')
        if isinstance(value, str):
            texts = [value]
        elif isinstance(value, list) and value and all(isinstance(item, str) for item in value):
            texts = value
        else:
            raise ValueError('input must be a string or nonempty list of strings; token ID arrays are not decoded')
        if submitted.get('dimensions', EMBED_DIM) != EMBED_DIM:
            raise ValueError(f'Only dimensions={EMBED_DIM} is supported')
        encoding = submitted.get('encoding_format', 'float')
        if encoding not in ('float', 'base64'):
            raise ValueError('encoding_format must be float or base64')
    except (ValueError, TypeError) as error:
        return error_response(record, started, submitted, submitted, str(error))
    queued = perf_counter()
    async with app.state.embedding_lock:
        acquired = perf_counter()
        record['queue_seconds'] = acquired - queued
        record['execution_started_at'] = now()
        try:
            vectors, lengths, norms = await asyncio.to_thread(embedding_work, texts)
            response = {'object': 'list', 'model': EMBED_MODEL,
                        'data': [{'object': 'embedding', 'index': index,
                                  'embedding': vector.tolist() if encoding == 'float' else
                                  base64.b64encode(vector.astype('<f4').tobytes()).decode('ascii')}
                                 for index, vector in enumerate(vectors)],
                        'usage': {'prompt_tokens': sum(lengths), 'total_tokens': sum(lengths)}}
            record.update(status=200, execution_seconds=perf_counter() - acquired,
                          execution_finished_at=now(), usage=response['usage'], token_lengths=lengths,
                          dimension=EMBED_DIM, normalized=True, vector_norms=norms.tolist(),
                          encoding_format=encoding, input_count=len(texts), truncation=False)
            finish_record(record, started, submitted, submitted, response)
            return JSONResponse(response, headers={'X-Request-ID': record['request_id']})
        except (ValueError, RuntimeError) as error:
            record['execution_seconds'] = perf_counter() - acquired
            record['execution_finished_at'] = now()
            return error_response(record, started, submitted, submitted, str(error),
                                  400 if isinstance(error, ValueError) else 500)
