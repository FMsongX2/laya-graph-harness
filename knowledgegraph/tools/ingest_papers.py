"""Prepare all verified paper text and build it incrementally in the shared Cognee dataset."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from cli import request

CORPUS = ROOT / "data/hackerton-papers"
JOB = ROOT / "state/paper-ingestion"
INPUTS = CORPUS / "cognee-inputs"


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def prepare():
    catalog = json.loads((CORPUS / "catalog.json").read_text())
    bibliography = json.loads(Path(catalog["source_manifest"]).read_text())
    metadata = {item["id"]: item for item in bibliography["records"]}
    papers = []
    unavailable = []
    INPUTS.mkdir(parents=True, exist_ok=True)
    for paper_id, record in sorted(catalog["records"].items()):
        if record["status"] != "verified_original":
            unavailable.append({"paper_id": paper_id, "title": record["title"], "status": record["status"]})
            continue
        pdf = CORPUS / record["pdf_path"]
        assert digest(pdf.read_bytes()) == record["pdf_sha256"], f"PDF hash mismatch: {paper_id}"
        pages = [json.loads(line) for line in (CORPUS / record["pages_path"]).read_text().splitlines()]
        assert [page["page"] for page in pages] == list(range(1, record["extraction"]["pages"] + 1))
        assert all(digest(page["text"].encode()) == page["text_sha256"] for page in pages)
        # The first paper was already built from the archived layout text.
        # Keep that immutable input; use PDF reading order for the remaining
        # papers so adjacent columns are not concatenated as one sentence.
        extraction_mode = "historical_layout" if paper_id == "P42" else "pdf_flow_v1"
        if extraction_mode == "pdf_flow_v1":
            flow_path = (CORPUS / record["pages_path"]).with_name("flow-pages-v1.jsonl")
            if flow_path.exists():
                pages = [json.loads(line) for line in flow_path.read_text().splitlines()]
            else:
                flow_pages = []
                for page in pages:
                    output = subprocess.run(["pdftotext", "-enc", "UTF-8", "-f", str(page["page"]),
                                             "-l", str(page["page"]), "-nopgbrk", str(pdf), "-"],
                                            capture_output=True, text=True, check=True).stdout
                    flow_pages.append({"page": page["page"], "text": output,
                                       "text_sha256": digest(output.encode()), "characters": len(output)})
                temporary = flow_path.with_suffix(".tmp")
                temporary.write_text("\n".join(json.dumps(page, ensure_ascii=False) for page in flow_pages) + "\n")
                temporary.replace(flow_path)
                pages = flow_pages
            assert [page["page"] for page in pages] == list(range(1, record["extraction"]["pages"] + 1))
            assert all(digest(page["text"].encode()) == page["text_sha256"] for page in pages)
        bibliographic = metadata[paper_id]
        slug = re.sub(r"[^a-z0-9]+", "-", record["title"].lower()).strip("-")[:100]
        folder = INPUTS if extraction_mode == "historical_layout" else INPUTS / "flow-v1"
        folder.mkdir(exist_ok=True)
        path = folder / (paper_id + "--" + slug + ".md")
        header = {
            "paper_id": paper_id, "title": record["title"], "authors": bibliographic.get("authors", []),
            "source_url": record["source_url"], "versioned_pdf_url": record.get("resolved_url"),
            "source_reported_date": bibliographic.get("date"), "publication_status": bibliographic.get("venue"),
            "original_pdf": str(pdf.resolve()), "pdf_sha256": record["pdf_sha256"],
            "pdf_pages": len(pages), "source_type": "academic_paper",
            "extraction_quality": record["extraction"]["quality_status"],
        }
        text = "---\n" + "\n".join(key + ": " + json.dumps(value, ensure_ascii=False) for key, value in header.items())
        text += "\n---\n\n# " + record["title"] + "\n"
        spans = []
        for page in pages:
            text += f"\n\n## PDF page {page['page']} of {len(pages)}\n\n"
            start = len(text)
            text += page["text"]
            spans.append({"pdf_page": page["page"], "start_char": start, "end_char": len(text),
                          "text_sha256": page["text_sha256"]})
        if path.exists():
            assert path.read_text() == text, f"Existing prepared paper changed: {paper_id}"
        else:
            path.write_text(text)
        save(path.with_suffix(".source.json"), {**header, "input_path": str(path),
             "input_sha256": digest(text.encode()), "page_spans": spans})
        papers.append({"paper_id": paper_id, "title": record["title"], "input_path": str(path),
                       "input_sha256": digest(text.encode()), "pdf_path": str(pdf.resolve()),
                       "pdf_sha256": record["pdf_sha256"], "pages": len(pages),
                       "category": bibliographic.get("category", "research"),
                       "extraction_mode": extraction_mode,
                       "characters": sum(len(page["text"]) for page in pages)})
    manifest = {"collection": catalog["collection_id"], "dataset": "domain_knowledge", "papers": papers,
                "paper_count": len(papers), "page_count": sum(paper["pages"] for paper in papers),
                "metadata_only_not_ingested": unavailable,
                "pdf_and_page_hashes_verified": True,
                "input_mode": "full PDF-page text with source sidecars; PDF reading order for new inputs; first verified layout input preserved",
                "visual_figures_interpreted": False}
    save(JOB / "manifest.json", manifest)
    return manifest


def completed(result):
    statuses = []
    def collect(value):
        if isinstance(value, dict):
            if "status" in value:
                statuses.append(value["status"])
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)
    collect(result)
    return bool(statuses) and all(status in {"PipelineRunCompleted", "PipelineRunAlreadyCompleted"} for status in statuses)


def reflect_native_state(progress, manifest, snapshot):
    by_name = {Path(paper["input_path"]).stem: paper for paper in manifest["papers"]}
    found = set()
    for row in snapshot["data"]:
        paper = by_name.get(row["name"])
        if not paper:
            continue
        paper_id = paper["paper_id"]
        found.add(paper_id)
        entry = progress["papers"].setdefault(paper_id, {"title": paper["title"], "pages": paper["pages"]})
        entry.update(registered=True, data_id=row["id"], token_count=row["token_count"])
        slot = (row.get("pipeline_status") or {}).get("cognify_pipeline", {}).get(snapshot["dataset_id"])
        if isinstance(slot, dict):
            slot = slot.get("status")
        if slot == "DATA_ITEM_PROCESSING_COMPLETED":
            entry["status"] = "complete"
            entry.setdefault("completed_at", now())
        elif entry.get("status") != "complete":
            entry["status"] = "pending_graph"
    progress["registered_papers"] = len(found)
    progress["completed_papers"] = sum(item.get("status") == "complete" for item in progress["papers"].values())
    progress["completed_pages"] = sum(item["pages"] for item in progress["papers"].values() if item.get("status") == "complete")
    progress["updated_at"] = now()
    return found


def verify(manifest):
    snapshot = request("/kg/datasets/" + manifest["dataset"] + "/data")
    rows = {row["name"]: row for row in snapshot["data"]}
    for paper in manifest["papers"]:
        row = rows[Path(paper["input_path"]).stem]
        assert row["token_count"] and row["token_count"] > 0, row
        slot = row["pipeline_status"]["cognify_pipeline"][snapshot["dataset_id"]]
        assert (slot.get("status") if isinstance(slot, dict) else slot) == "DATA_ITEM_PROCESSING_COMPLETED"
    save(JOB / "results/dataset-final.json", snapshot)
    queries = {
        "rank-fusion": "How does Reciprocal Rank Fusion combine ranked retrieval lists?",
        "graph-retrieval": "When are graph community summaries useful for global questions across a document corpus?",
        "temporal": "How can temporal graph retrieval handle time conflicts and temporal leakage?",
        "embeddings": "What multilingual dense sparse and multi-vector retrieval capabilities does BGE-M3 provide?",
        "topics": "What methods support topic modeling and discovering new clusters with limited labels?",
        "adaptive-retrieval": "When should retrieval be skipped or adapted according to question complexity?",
        "citations": "How can retrieval and source citations support faithful scientific claims?",
        "policy": "What conditions matter when studying policy diffusion and industrial policy evidence?",
    }
    probes = []
    for name, query in queries.items():
        result = request("/kg/search", {"query": query, "datasets": [manifest["dataset"]],
                                        "mode": "HYBRID_COMPLETION", "top_k": 5})
        save(JOB / "results" / ("verify-" + name + ".json"), result)
        assert result["answer_generated"] is False and result["results"], name
        objects = [item.get("objects_result") or {} for item in result["results"]]
        sources = [chunk["payload"].get("paper_source") for obj in objects for chunk in obj.get("chunks", [])]
        sources = [source for source in sources if source]
        assert sources, f"No original-paper reference returned: {name}"
        probes.append({"name": name, "papers": sorted(set(source["paper_id"] for source in sources)),
                       "entities": sum(len(obj.get("entities", [])) for obj in objects),
                       "pages_located": sum(bool(source["pdf_pages"]) for source in sources)})
    report = {"verified_at": now(), "dataset": manifest["dataset"], "papers": len(manifest["papers"]),
              "source_pages": manifest["page_count"], "all_documents_native_pipeline_complete": True,
              "queries": probes, "full_paper_corpus_indexed": True,
              "limits": "Text-layer indexing and sampled retrieval checks; figures and equation semantics were not visually interpreted; extracted facts are not all manually reviewed."}
    save(JOB / "verification.json", report)
    return report


def run(manifest):
    JOB.mkdir(parents=True, exist_ok=True)
    with (JOB / "worker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        progress_path = JOB / "progress.json"
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {
            "created_at": now(), "dataset": manifest["dataset"], "papers": {}}
        progress.update(status="registering", worker_pid=os.getpid(), total_papers=manifest["paper_count"],
                        total_pages=manifest["page_count"], updated_at=now())
        progress.pop("error", None)
        save(progress_path, progress)
        pending = sorted(manifest["papers"], key=lambda item: (item["pages"], item["paper_id"]))
        try:
            # Register the complete corpus first. Cognee's one-document batches
            # then checkpoint completed documents in the shared dataset.
            for paper in pending:
                paper_id = paper["paper_id"]
                if progress["papers"].get(paper_id, {}).get("status") == "complete":
                    continue
                progress["current_paper"] = paper_id
                progress["papers"][paper_id] = {"title": paper["title"], "pages": paper["pages"],
                                               "status": "registering", "started_at": now()}
                progress["updated_at"] = now()
                save(progress_path, progress)
                tags = ["research-papers", paper["category"]]
                added = request("/kg/add", {"files": [paper["input_path"]], "dataset": manifest["dataset"],
                                             "domains": tags}, timeout=86400)
                save(JOB / "results" / (paper_id + "-add.json"), added)
                assert completed(added), added
                progress["papers"][paper_id].update(status="pending_graph", registered=True)
                progress["registered_papers"] = sum(item.get("registered") or item.get("status") == "complete"
                                                    for item in progress["papers"].values())
                progress["updated_at"] = now()
                save(progress_path, progress)
                print(json.dumps({"paper": paper_id, "status": "registered"}), flush=True)
            snapshot = request("/kg/datasets/" + manifest["dataset"] + "/data")
            found = reflect_native_state(progress, manifest, snapshot)
            assert found == {paper["paper_id"] for paper in manifest["papers"]}, found
            save(JOB / "results/registered-corpus.json", snapshot)
            save(progress_path, progress)
            parameters = {"dataset": manifest["dataset"], "chunk_size": 1024,
                          "chunks_per_batch": 8, "data_per_batch": 1, "paper_pages": True}
            try:
                dry_run = request("/kg/build", {**parameters, "dry_run": True}, timeout=172800)
            except RuntimeError as error:
                # Cognee's optional estimator uses TikToken with disallowed
                # special tokens. P41 quotes a model delimiter as paper data.
                # Keep the source intact; exact BGE chunk checks already passed.
                if "disallowed special token" not in str(error):
                    raise
                dry_run = {"result": {"unavailable": str(error),
                            "source_preserved": True,
                            "actual_embedding_chunk_check": str(ROOT / "tests/paper-chunking-result.json")}}
            save(JOB / "results/corpus-dry-run.json", dry_run)
            progress.update(status="building", build_started_at=now(), dry_run=dry_run["result"], updated_at=now())
            progress.pop("current_paper", None)
            save(progress_path, progress)
            print(json.dumps({"status": "building", "registered": len(found), "dry_run": dry_run["result"]}), flush=True)
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(request, "/kg/build", parameters, 172800)
                last_count = progress["completed_papers"]
                while not future.done():
                    try:
                        snapshot = request("/kg/datasets/" + manifest["dataset"] + "/progress", timeout=30)
                        reflect_native_state(progress, manifest, snapshot)
                        save(progress_path, progress)
                        if progress["completed_papers"] != last_count:
                            last_count = progress["completed_papers"]
                            print(json.dumps({"completed": last_count, "total": progress["total_papers"]}), flush=True)
                    except (RuntimeError, OSError) as error:
                        progress["last_progress_read_error"] = str(error)
                        save(progress_path, progress)
                    time.sleep(15)
                built = future.result()
            save(JOB / "results/corpus-build.json", built)
            assert completed(built), built
            snapshot = request("/kg/datasets/" + manifest["dataset"] + "/data")
            reflect_native_state(progress, manifest, snapshot)
            assert progress["completed_papers"] == manifest["paper_count"], progress
            progress.update(status="verifying", updated_at=now())
            save(progress_path, progress)
            verification = verify(manifest)
            progress["verification"] = str(JOB / "verification.json")
            progress.update(status="complete", completed_at=now(), updated_at=now())
            progress.pop("current_paper", None)
            save(progress_path, progress)
            print(json.dumps({"status": "complete", "papers": verification["papers"], "pages": verification["source_pages"]}), flush=True)
        except Exception as error:
            progress.update(status="failed", error=str(error), updated_at=now())
            save(progress_path, progress)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    manifest = prepare()
    print(json.dumps({"papers": manifest["paper_count"], "pages": manifest["page_count"],
                      "metadata_only": len(manifest["metadata_only_not_ingested"])}), flush=True)
    if not args.prepare_only:
        run(manifest)
