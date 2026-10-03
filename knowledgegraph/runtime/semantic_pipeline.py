"""Build a qualified semantic graph from accepted assertions with Cognee, no re-inference."""
from pathlib import Path
import json
from uuid import uuid5
import cognee
from cognee.modules.pipelines import Task
from cognee.tasks.documents import classify_documents, extract_chunks_from_documents
from cognee.tasks.graph.extract_graph_from_data import integrate_chunk_graphs
from cognee.shared.data_models import KnowledgeGraph, Node, Edge
from cognee.tasks.summarization.models import TextSummary
from cognee.tasks.storage import add_data_points
from cognee.tasks.provenance.record_provenance import record_provenance
from card_pipeline import WholeCardChunker
from semantic_sources import attach_semantic_sources, indexed_text_matches

ROOT = Path(__file__).resolve().parents[1]


def assertion_graph(assertion):
    nodes = [];edges = []
    def node(key, name, kind, description):
        nodes.append(Node(id=key, name=name, type=kind, description=description));return key
    def edge(left, right, relation, text):
        edges.append(Edge(source_node_id=left, target_node_id=right, relationship_name=relation, description=text))
    source = node("source", assertion["source"]["name"], assertion["source"]["type"], assertion["source"]["name"])
    target = node("target", assertion["target"]["name"], assertion["target"]["type"], assertion["target"]["name"])
    statement = node("assertion", assertion["id"] + " @" + assertion["pdf_sha256"][:12], "SourceAssertion", assertion["retrieval_text"])
    paper = node("paper", assertion["paper_id"] + ": " + assertion["paper_title"], "SourceDocument", assertion["source_url"])
    # Reify the statement instead of collapsing multiple experiments/conditions
    # into an unconditional source->target edge. Direction stays explicit.
    edge(source, statement, "has_assertion", assertion["text"])
    edge(statement, target, assertion["relation"], assertion["text"])
    edge(statement, paper, "source_reviewed_against", "Source review and exact quote matching; not scientific truth certification")
    for i, condition in enumerate(assertion["conditions"]):
        constraint = node(f"condition-{i}", assertion["id"] + " scope " + str(i), "ApplicabilityCondition", condition)
        edge(statement, constraint, "under_condition", condition)
    for i, evidence in enumerate(assertion["evidence"]):
        reference = node(f"evidence-{i}", f"{assertion['id']} PDF p{evidence['page']} evidence {i}", "SourceEvidence", evidence["quote"])
        edge(statement, reference, "supported_by", evidence["quote"])
        edge(reference, paper, "sourced_from", f"PDF page {evidence['page']}")
    for role in ["source", "target"]:
        for i, alias in enumerate(assertion[role].get("aliases", [])):
            alias_node = node(f"alias-{role}-{i}", assertion[role]["type"] + " alias: " + alias, "ReviewedAlias", alias)
            edge(role, alias_node, "has_source_reviewed_alias", f"{alias} refers to {assertion[role]['name']} in the reviewed source")
    return KnowledgeGraph(nodes=nodes, edges=edges)


async def attach_assertions(chunks, ctx=None):
    registry = json.loads((ROOT / "state/semantic-ingestion/assertions.json").read_text())
    records = {Path(item["markdown_path"]).stem: item for item in registry["assertions"]}
    graphs = [];summaries = []
    for chunk in chunks:
        assertion = json.loads(Path(records[chunk.document_name]["json_path"]).read_text())
        if not indexed_text_matches(chunk.text, assertion):
            raise ValueError("Indexed assertion text does not match the approved record: " + chunk.document_name)
        graphs.append(assertion_graph(assertion))
        summaries.append(TextSummary(id=uuid5(chunk.id, "TextSummary"), text=assertion["retrieval_text"],
            made_from=chunk, source_chunk_id=str(chunk.id), belongs_to_set=chunk.belongs_to_set))
    await integrate_chunk_graphs(chunks, graphs, KnowledgeGraph, None, ctx=ctx,
        pipeline_name="custom_pipeline", task_name="source_reviewed_semantic_assertions")
    return summaries


async def index_assertions(dataset, data_ids, *, graph_only=False, force_rebuild=False):
    available = await cognee.datasets.list_datasets(); selected = next(item for item in available if item.name == dataset)
    rows = await cognee.datasets.list_data(selected.id);data = [row for row in rows if str(row.id) in data_ids]
    if len(data) != len(data_ids):raise ValueError("Semantic assertion IDs do not match the requested dataset")
    tasks = [Task(classify_documents, needs_llm=False),
             Task(extract_chunks_from_documents, max_chunk_size=8190, chunker=WholeCardChunker, needs_llm=False),
             Task(attach_assertions, needs_llm=False, task_config={"batch_size": 4}),
             Task(add_data_points, graph_only=graph_only, needs_llm=False, task_config={"batch_size": 4}),
             Task(record_provenance, needs_llm=False, task_config={"batch_size": 4})]
    return await cognee.run_custom_pipeline(tasks=tasks, data=data, dataset=dataset,
        incremental_loading=not force_rebuild, data_per_batch=1, pipeline_name="custom_pipeline")
