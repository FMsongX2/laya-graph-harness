"""Cognee pipelines for deterministic card graphs and original-source vectors."""
from pathlib import Path
import json
from uuid import uuid5

import cognee
from cognee.modules.chunking.Chunker import Chunker
from cognee.modules.chunking.chunk_id import chunk_content_hash, content_chunk_id
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.pipelines import Task
from cognee.shared.data_models import KnowledgeGraph, Node, Edge
from cognee.tasks.documents import classify_documents, extract_chunks_from_documents
from cognee.tasks.graph.extract_graph_from_data import integrate_chunk_graphs
from cognee.tasks.storage import add_data_points
from cognee.tasks.provenance.record_provenance import record_provenance
from cognee.tasks.summarization.models import TextSummary
from paper_chunker import PaperPageChunker, tokenizer

ROOT = Path(__file__).resolve().parents[1]


class WholeCardChunker(Chunker):
    chunker_id = "knowledge_card_atomic_v1"

    async def read(self):
        async for text in self.get_text():
            count = len(tokenizer().encode(text, add_special_tokens=False).ids)
            if count > 8190:
                raise ValueError("Knowledge card exceeds the BGE input limit")
            digest = chunk_content_hash(text)
            yield DocumentChunk(id=content_chunk_id(str(self.document.id), digest, self.chunk_index),
                chunker_id=self.chunker_id, text=text, chunk_size=count, max_chunk_tokens=8190,
                content_hash=digest, is_part_of=self.document, contains=[],
                chunk_index=self.chunk_index, cut_type="knowledge_card",
                document_id=str(self.document.id), document_name=self.document.name,
                importance_weight=self.document.importance_weight, metadata={"index_fields": ["text"]})
            self.chunk_index += 1


def graph_from_card(card):
    nodes = []; edges = []
    def node(key, name, kind, description):
        nodes.append(Node(id=key, name=name, type=kind, description=description))
        return key
    def edge(source, target, kind, description):
        edges.append(Edge(source_node_id=source, target_node_id=target,
                          relationship_name=kind, description=description))
    root = node("card", card["card_id"], "KnowledgeCard", card["text"])
    method = node("subject", card["subject"], "MethodOrConcept", card["title"])
    paper = node("paper", card["paper_id"] + ": " + card["paper_title"], "Paper", card["source_url"])
    edge(root, method, "describes", card["title"])
    edge(root, paper, "derived_from", "The card is derived from " + card["paper_title"])
    kinds = {"problem": ("Task", "addresses"), "mechanism": ("Mechanism", "works_by"),
             "preprocessing": ("Preprocessing", "uses_preprocessing"), "applicability": ("Condition", "applies_when"),
             "requirement": ("Condition", "requires"), "alternative": ("Alternative", "has_alternative"),
             "evaluation": ("ReportedFinding", "has_reported_result"), "limitation": ("Limitation", "has_limitation")}
    for i, claim in enumerate(card["claims"]):
        kind, relation = kinds[claim["kind"]]
        claim_node = node(f"claim-{i}", claim["text"] + " [" + card["card_id"] + "]", kind, claim["text"])
        edge(method, claim_node, relation, claim["text"])
        edge(root, claim_node, "includes_claim", claim["text"])
        for j, evidence in enumerate(claim["evidence"]):
            locator = "PDF p" + str(evidence["page"]) if evidence.get("page") else evidence["locator"]
            evidence_node = node(f"evidence-{i}-{j}", f"{card['card_id']} {locator} evidence {i}-{j}",
                                 "Evidence", evidence["quote"])
            edge(claim_node, evidence_node, "supported_by", evidence["quote"])
            edge(evidence_node, paper, "sourced_from", locator)
    return KnowledgeGraph(nodes=nodes, edges=edges)


async def attach_card_graphs(chunks, ctx=None):
    registry = json.loads((ROOT / "state/card-ingestion/cards.json").read_text())
    cards = {Path(item["markdown_path"]).stem: item for item in registry["cards"]}
    graphs = []
    summaries = []
    for chunk in chunks:
        record = cards[chunk.document_name]
        card = json.loads(Path(record["json_path"]).read_text())
        graphs.append(graph_from_card(card))
        summaries.append(TextSummary(id=uuid5(chunk.id, "TextSummary"), text=card["text"],
                          made_from=chunk, source_chunk_id=str(chunk.id), belongs_to_set=chunk.belongs_to_set))
    await integrate_chunk_graphs(chunks, graphs, KnowledgeGraph, None, ctx=ctx,
                                pipeline_name="custom_pipeline", task_name="validated_card_graph")
    return summaries


async def build_index(dataset, cards=False, data_ids=None):
    config = json.loads((ROOT / "config/settings.json").read_text())
    tasks = [Task(classify_documents, needs_llm=False),
             Task(extract_chunks_from_documents, max_chunk_size=8190 if cards else 1024,
                  chunker=WholeCardChunker if cards else PaperPageChunker, needs_llm=False)]
    if cards:
        tasks.append(Task(attach_card_graphs, needs_llm=False, task_config={"batch_size": 4}))
    tasks += [Task(add_data_points, needs_llm=False, task_config={"batch_size": 16}),
              Task(record_provenance, needs_llm=False, task_config={"batch_size": 16})]
    data = None
    if data_ids:
        available = await cognee.datasets.list_datasets()
        selected = next(item for item in available if item.name == dataset)
        rows = await cognee.datasets.list_data(selected.id)
        data = [row for row in rows if str(row.id) in data_ids]
        if len(data) != len(data_ids):
            raise ValueError("Card input data IDs do not match the requested dataset")
    return await cognee.run_custom_pipeline(tasks=tasks, data=data, dataset=dataset,
                 incremental_loading=True, data_per_batch=1, pipeline_name="custom_pipeline")


def attach_card_sources(result):
    registry_path = ROOT / "state/card-ingestion/cards.json"
    if not registry_path.exists():
        return result
    records = json.loads(registry_path.read_text())["cards"]
    cards = {Path(item["markdown_path"]).stem: item for item in records}
    def walk(value):
        if isinstance(value, dict):
            name = value.get("document_name")
            if name in cards:
                card = json.loads(Path(cards[name]["json_path"]).read_text())
                value["knowledge_card"] = card
            for child in list(value.values()):
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(result)
    return result
