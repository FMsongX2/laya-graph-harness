"""Cognee 상태와 모델 접점을 이 작업 폴더에 고정한다."""
from pathlib import Path
import json
try: from . import portable
except ImportError: import portable

ROOT = Path(__file__).resolve().parents[1]


def environment():
    """상속된 API 인증·프록시 설정 없이 로컬 실행 환경을 만든다."""
    config = json.loads((ROOT / "config/settings.json").read_text())
    env = portable.passthrough_environment(("PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TERM"))
    directories = {
        "DATA_ROOT_DIRECTORY": "state/data",
        "SYSTEM_ROOT_DIRECTORY": "state/system",
        "CACHE_ROOT_DIRECTORY": "state/cache",
        "COGNEE_LOGS_DIR": "logs/cognee",
        "COGNEE_REPOS_DIR": "state/repos",
        "XDG_CACHE_HOME": "runtime/cache/xdg",
        "HF_HOME": "runtime/cache/hf",
        "TORCH_HOME": "runtime/cache/torch",
        "TIKTOKEN_CACHE_DIR": "runtime/cache/tiktoken",
        "TMPDIR": "runtime/tmp",
        "NUMBA_CACHE_DIR": "runtime/cache/numba",
        "MPLCONFIGDIR": "runtime/cache/matplotlib",
    }
    for key, name in directories.items():
        path = ROOT / name
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env.update(portable.temporary_environment(ROOT / "runtime/tmp"))
    # Cognee resolves the embedding tokenizer by its HF repo name. Give it an
    # offline cache view of the exact existing snapshot, without touching it.
    snapshot = Path(config["embedding"]["snapshot"]).resolve()
    cache = ROOT / "runtime/cache/hf/hub"
    model_cache = cache / ("models--" + config["embedding"]["model"].replace("/", "--"))
    cached_snapshot = model_cache / "snapshots" / snapshot.name
    cached_snapshot.parent.mkdir(parents=True, exist_ok=True)
    if not cached_snapshot.exists():
        portable.link_directory(cached_snapshot, snapshot)
    if cached_snapshot.resolve() != snapshot:
        raise RuntimeError("Tokenizer cache points to a different model snapshot")
    (model_cache / "refs").mkdir(exist_ok=True)
    (model_cache / "refs/main").write_text(snapshot.name)
    env["HF_HUB_CACHE"] = str(cache)
    endpoint = f"http://127.0.0.1:{config['server']['port']}/v1"
    env.update({
        "PYTHON_DOTENV_DISABLED": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
        "TELEMETRY_DISABLED": "1", "COGNEE_TRACING_ENABLED": "false",
        "LITELLM_LOCAL_MODEL_COST_MAP": "true", "LITELLM_LOG": "ERROR",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "CACHING": "false", "AUTO_FEEDBACK": "false", "USAGE_LOGGING": "false",
        "PERSONALIZATION_ENABLED": "false", "DEFAULT_FEEDBACK_INFLUENCE": "0",
        "PROVENANCE_TRACKING": str(config['cognee']['provenance_tracking']).lower(),
        "GRAPH_EXTRACTOR": "llm", "GLINER_AUTO_INSTALL": "false",
        "GRAPH_PROMPT_PATH": str(ROOT / config['cognee']['graph_prompt_path']),
        "ENABLE_BACKEND_ACCESS_CONTROL": "true",
        "GRAPH_DATABASE_PROVIDER": config.get("graph", {}).get("provider", "ladybug"),
        "GRAPH_DATASET_DATABASE_HANDLER": config.get("graph", {}).get("dataset_handler", "ladybug"),
        "KUZU_NUM_THREADS": "4", "KUZU_BUFFER_POOL_SIZE": "536870912",
        "KUZU_MAX_DB_SIZE": "137438953472",
        "VECTOR_DB_PROVIDER": "lancedb", "VECTOR_DATASET_DATABASE_HANDLER": "lancedb",
        "DB_PROVIDER": "sqlite",
        "LLM_PROVIDER": "openai", "LLM_MODEL": "openai/" + config['chat']['model'],
        "LLM_ENDPOINT": endpoint, "LLM_API_KEY": "local-knowledge-graph",
        "LLM_TEMPERATURE": str(config['chat']['temperature']),
        "LLM_MAX_COMPLETION_TOKENS": str(config['chat']['max_output_tokens']),
        "LLM_RATE_LIMIT_ENABLED": "false",
        "EMBEDDING_PROVIDER": "openai_compatible", "EMBEDDING_MODEL": config['embedding']['model'],
        "EMBEDDING_ENDPOINT": endpoint, "EMBEDDING_API_KEY": "local-knowledge-graph",
        "EMBEDDING_DIMENSIONS": str(config['embedding']['dimensions']),
        "EMBEDDING_MAX_COMPLETION_TOKENS": str(config['embedding']['max_tokens']),
        "NO_PROXY": "localhost,127.0.0.1,::1",
    })
    if config.get("graph", {}).get("provider") == "neo4j":
        graph = config["graph"]
        secret = json.loads((ROOT / graph["encryption_key_file"]).read_text())
        env.update({"NEO4J_ENCRYPTION_KEY": secret["encryption_key"],
                    "NEO4J_COMMUNITY_IMAGE": graph["image"],
                    "NEO4J_COMMUNITY_MAX_CONTAINERS": str(graph["max_running_containers"]),
                    "NEO4J_COMMUNITY_STARTUP_TIMEOUT": "180"})
    return env
