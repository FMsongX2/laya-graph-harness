"""Cognee SDK에 수집·구축·근거 조회만 연결한다. 자체 검색 엔진은 구현하지 않는다."""
import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cognee
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from paper_sources import attach_paper_sources
from paper_chunker import PaperPageChunker
from card_pipeline import build_index, attach_card_sources
from semantic_pipeline import index_assertions, attach_semantic_sources

CONFIG = json.loads((Path(__file__).resolve().parents[1] / "config/settings.json").read_text())

router = APIRouter(prefix="/kg")


def plain(value: Any):
    """SDK의 UUID·Edge·Pydantic 반환값을 JSON 표현으로 바꾼다."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [plain(item) for item in value]
    if hasattr(value, "model_dump"):
        return plain(value.model_dump(mode="python"))
    if hasattr(value, "to_json"):
        return plain(value.to_json())
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    return str(value)


@asynccontextmanager
async def operation(request: Request, name: str):
    """한 로컬 저장소의 SDK 작업을 직렬화하고 실패를 호출자에게 전달한다."""
    async with request.app.state.knowledge_lock:
        request.app.state.knowledge_operation = name
        try:
            yield
        except asyncio.CancelledError:
            raise
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail={"type": type(exc).__name__, "message": str(exc)}) from exc
        finally:
            request.app.state.knowledge_operation = None


class AddRequest(BaseModel):
    files: list[str] = Field(default_factory=list)
    text: str | None = None
    dataset: str = CONFIG["knowledge"]["evidence_dataset"]
    domains: list[str] = Field(default_factory=list)


class BuildRequest(BaseModel):
    dataset: str = CONFIG["default_dataset"]
    chunk_size: int = Field(default=512, ge=128, le=4096)
    dry_run: bool = False
    chunks_per_batch: int = Field(default=8, ge=1, le=128)
    data_per_batch: int = Field(default=1, ge=1, le=20)
    paper_pages: bool = False


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    datasets: list[str] = Field(default_factory=lambda: [CONFIG["default_dataset"]])
    domains: list[str] = Field(default_factory=list)
    mode: str = "HYBRID_COMPLETION"
    top_k: int = Field(default=10, ge=1, le=100)


class IndexRequest(BaseModel):
    dataset: str
    cards: bool = False
    data_ids: list[str] = Field(default_factory=list)


class SemanticIndexRequest(BaseModel):
    data_ids: list[str] = Field(min_length=1)
    graph_only: bool = False
    force_rebuild: bool = False


@router.get("/datasets")
async def datasets(request: Request):
    async with operation(request, "datasets"):
        return {"datasets": plain(await cognee.datasets.list_datasets())}


@router.get("/graph-ready")
async def graph_ready(request: Request):
    """Warm only the authorized semantic container through Cognee's own lifecycle."""
    from cognee.context_global_variables import set_database_global_context_variables
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.modules.data.methods import get_authorized_existing_datasets
    from cognee.modules.users.methods import get_default_user
    async with operation(request, "graph-ready"):
        user = await get_default_user()
        datasets = await get_authorized_existing_datasets([CONFIG['knowledge']['semantic_dataset']], "read", user)
        if len(datasets) != 1:
            raise HTTPException(status_code=404, detail="Semantic dataset is unavailable")
        dataset = datasets[0]
        async with set_database_global_context_variables(dataset.id, dataset.owner_id):
            engine = await get_graph_engine()
            await engine.query("RETURN 1 AS ready")
        return {"ready": True, "dataset": dataset.name, "graph_data_writes": False}


async def dataset_rows(dataset_name: str):
    available = await cognee.datasets.list_datasets()
    matches = [item for item in available if item.name == dataset_name]
    if len(matches) != 1:
        raise HTTPException(status_code=404, detail="Dataset name is missing or ambiguous")
    rows = await cognee.datasets.list_data(matches[0].id)
    return {"dataset": dataset_name, "dataset_id": str(matches[0].id),
            "data": [{"id": str(row.id), "name": row.name, "token_count": row.token_count,
                      "original_data_location": row.original_data_location,
                      "raw_data_location": row.raw_data_location,
                      "pipeline_status": plain(row.pipeline_status)} for row in rows]}


@router.get("/datasets/{dataset_name}/data")
async def dataset_data(dataset_name: str, request: Request):
    async with operation(request, "list-data"):
        return await dataset_rows(dataset_name)


@router.get("/datasets/{dataset_name}/progress")
async def dataset_progress(dataset_name: str):
    # Read relational metadata only; taking the graph operation lock would make
    # progress unavailable throughout a long build. No graph engine is opened.
    return await dataset_rows(dataset_name)


@router.get("/datasets/{dataset_name}/graph-integrity")
async def graph_integrity(dataset_name: str, request: Request):
    """Read the authorized physical graph and compare registered completed assertions."""
    if dataset_name != CONFIG["knowledge"]["semantic_dataset"]:
        raise HTTPException(status_code=400, detail="This assertion check is defined only for the semantic dataset")
    from cognee.api.v1.visualize.visualize import fetch_visualization_data
    async with operation(request, "graph-integrity"):
        metadata = await dataset_rows(dataset_name)
        graph_data, _ = await fetch_visualization_data(dataset=dataset_name, full=True, include_session_events=False)
        nodes, edges = graph_data
        names = {str(properties.get("name", "")).casefold().strip() for _, properties in nodes}
        root = Path(__file__).resolve().parents[1]
        registry = json.loads((root / "state/semantic-ingestion/assertions.json").read_text())["assertions"]
        completed = {row["name"] for row in metadata["data"] if
                     row["pipeline_status"].get("custom_pipeline", {}).get(metadata["dataset_id"]) == "DATA_ITEM_PROCESSING_COMPLETED"}
        expected = {}
        for record in registry:
            if Path(record["markdown_path"]).stem in completed:
                assertion = json.loads(Path(record["json_path"]).read_text())
                expected[assertion["id"]] = (assertion["id"] + " @" + assertion["pdf_sha256"][:12]).casefold()
        missing = [identity for identity, name in expected.items() if name not in names]
        return {"dataset": dataset_name, "physical_node_count": len(nodes), "physical_edge_count": len(edges),
                "completed_registered_assertions": len(expected), "present_assertion_nodes": len(expected) - len(missing),
                "missing_assertion_ids": missing, "assertion_node_coverage_passed": not missing,
                "scientific_truth_certified": False}


@router.post("/add")
async def add(body: AddRequest, request: Request):
    items = []
    for name in body.files:
        path = Path(name).expanduser().resolve()
        if not path.is_file():
            raise HTTPException(status_code=400, detail=f"Not a file: {path}")
        items.append(str(path))
    if body.text and body.text.strip():
        items.append(body.text)
    if not items:
        raise HTTPException(status_code=400, detail="Provide files or text")
    async with operation(request, "add"):
        result = await cognee.add(items, dataset_name=body.dataset, node_set=body.domains or None)
        return {"dataset": body.dataset, "items_added": len(items), "graph_built": False, "result": plain(result)}


@router.post("/build")
async def build(body: BuildRequest, request: Request):
    async with operation(request, "dry-run" if body.dry_run else "build"):
        if body.dataset == CONFIG["knowledge"].get("semantic_dataset"):
            if body.dry_run:
                return {"dataset": body.dataset, "additional_llm_calls": 0,
                        "note": "Only independently accepted semantic assertion records may be indexed."}
            rows = (await dataset_rows(body.dataset))["data"]
            result = await index_assertions(body.dataset, [row["id"] for row in rows])
            return {"dataset": body.dataset, "result": plain(result), "additional_llm_calls": 0}
        if body.dataset in {CONFIG["knowledge"]["cards_dataset"], CONFIG["knowledge"]["evidence_dataset"]}:
            if body.dry_run:
                return {"dataset": body.dataset, "dry_run": True, "llm_calls": 0,
                        "note": "Deterministic index; card generation is performed by the card worker."}
            result = await build_index(body.dataset, cards=body.dataset == CONFIG["knowledge"]["cards_dataset"])
            return {"dataset": body.dataset, "dry_run": False, "result": plain(result), "llm_calls": 0}
        chunker = {"chunker": PaperPageChunker} if body.paper_pages else {}
        result = await cognee.cognify(datasets=[body.dataset], chunk_size=body.chunk_size,
                                     incremental_loading=True, extractor="llm", dry_run=body.dry_run,
                                     chunks_per_batch=body.chunks_per_batch, data_per_batch=body.data_per_batch,
                                     **chunker,
                                     raise_on_error=True)
        return {"dataset": body.dataset, "dry_run": body.dry_run, "result": plain(result)}


@router.post("/index")
async def index(body: IndexRequest, request: Request):
    if body.cards and CONFIG["knowledge"]["strategy"] != "knowledge_cards":
        raise HTTPException(status_code=409, detail="Knowledge-card indexing is inactive for the selected fulltext strategy")
    expected = CONFIG["knowledge"]["cards_dataset"] if body.cards else CONFIG["knowledge"]["evidence_dataset"]
    if body.dataset != expected:
        raise HTTPException(status_code=400, detail="Index dataset does not match its declared layer")
    async with operation(request, "card-index" if body.cards else "evidence-index"):
        result = await build_index(body.dataset, cards=body.cards, data_ids=body.data_ids or None)
        return {"dataset": body.dataset, "result": plain(result), "llm_calls": 0}


@router.post("/search")
async def search(body: SearchRequest, request: Request):
    if body.mode not in {"HYBRID_COMPLETION", "GRAPH_COMPLETION", "CHUNKS"}:
        raise HTTPException(status_code=400, detail="Unsupported context-only query mode")
    async with operation(request, "search"):
        result = await cognee.search(query_text=body.query, query_type=getattr(cognee.SearchType, body.mode),
                                     datasets=body.datasets, node_name=body.domains or None,
                                     only_context=True, verbose=True, include_references=True,
                                     top_k=body.top_k)
        return {"query": body.query, "requested_mode": body.mode, "answer_generated": False,
                "datasets": body.datasets, "results": attach_semantic_sources(attach_card_sources(attach_paper_sources(plain(result))))}


@router.post("/semantic-index")
async def semantic_index(body: SemanticIndexRequest, request: Request):
    dataset = CONFIG["knowledge"]["semantic_dataset"]
    if body.graph_only and not body.force_rebuild:
        raise HTTPException(status_code=400, detail="Graph-only restoration must explicitly force reconstruction of selected approved data")
    async with operation(request, "semantic-index"):
        result = await index_assertions(dataset, body.data_ids, graph_only=body.graph_only, force_rebuild=body.force_rebuild)
        return {"dataset": dataset, "result": plain(result), "additional_llm_calls": 0,
                "graph_only": body.graph_only, "force_rebuild": body.force_rebuild,
                "assertion_generation": "independent source-reading and source-review sessions"}
