"""Index all complete source text with Cognee, without generated semantic claims."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import sys
import fcntl
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from cli import request
CONFIG = json.loads((ROOT / "config/settings.json").read_text())
JOB = ROOT / "state/fulltext-ingestion"


def save(name, value):
    JOB.mkdir(parents=True, exist_ok=True)
    path = JOB / name; temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def run():
    JOB.mkdir(parents=True, exist_ok=True)
    with (JOB / "worker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = json.loads((ROOT / "state/paper-ingestion/manifest.json").read_text())
        dataset = CONFIG["knowledge"]["evidence_dataset"]
        state = {"strategy": "fulltext", "dataset": dataset, "status": "registering",
                 "worker_pid": os.getpid(), "total_papers": len(manifest["papers"]),
                 "total_pages": manifest["page_count"], "updated_at": datetime.now(timezone.utc).isoformat()}
        save("progress.json", state)
        try:
            result = request("/kg/add", {"files": [paper["input_path"] for paper in manifest["papers"]],
                             "dataset": dataset, "domains": ["original-sources"]}, timeout=3600)
            save("registration.json", result)
            state.update(status="indexing", registered_papers=len(manifest["papers"]));save("progress.json", state)
            result = request("/kg/index", {"dataset": dataset, "cards": False}, timeout=7200)
            save("index-result.json", result)
            verify()
        except Exception as error:
            state.update(status="failed", error=str(error), updated_at=datetime.now(timezone.utc).isoformat())
            save("progress.json", state)
            raise


def verify():
    manifest = json.loads((ROOT / "state/paper-ingestion/manifest.json").read_text())
    dataset = CONFIG["knowledge"]["evidence_dataset"]
    rows = request("/kg/datasets/" + dataset + "/data")
    by_name = {row["name"]: row for row in rows["data"]}
    for paper in manifest["papers"]:
        row = by_name[Path(paper["input_path"]).stem]
        slot = row["pipeline_status"]["custom_pipeline"][rows["dataset_id"]]
        assert (slot.get("status") if isinstance(slot, dict) else slot) == "DATA_ITEM_PROCESSING_COMPLETED", row
        assert row["token_count"] > 0, row
    probes = []
    for query in ["Reciprocal Rank Fusion constant k and combining rankings", "BGE-M3 multilingual embedding dense sparse multi-vector",
                  "GraphRAG community summaries global questions", "temporal conflicts knowledge graph retrieval"]:
        result = request("/kg/search", {"query": query, "datasets": [dataset], "mode": "CHUNKS", "top_k": 5})
        assert result["results"] and result["answer_generated"] is False
        references = []
        def collect(value):
            if isinstance(value, dict):
                if "paper_source" in value: references.append(value["paper_source"])
                for child in value.values():collect(child)
            elif isinstance(value, list):
                for child in value:collect(child)
        collect(result)
        assert references, query
        probes.append({"query": query, "source_papers": sorted(set(ref["paper_id"] for ref in references)),
                       "page_references": sum(bool(ref["pdf_pages"]) for ref in references)})
        save("query-" + str(len(probes)) + ".json", result)
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "strategy": "fulltext", "dataset": dataset,
              "papers": len(manifest["papers"]), "source_pages": manifest["page_count"],
              "all_sources_indexed": True, "indexing_llm_calls": 0, "queries": probes,
              "graph_relations": "source document and original text-chunk provenance; generated semantic relations are not active",
              "limits": "PDF text-layer indexing; figures and equation semantics have not been visually interpreted."}
    save("verification.json", report)
    save("progress.json", {"status": "complete", "strategy": "fulltext", "dataset": dataset,
         "total_papers": len(manifest["papers"]), "completed_papers": len(manifest["papers"]),
         "total_pages": manifest["page_count"], "completed_pages": manifest["page_count"],
         "updated_at": report["checked_at"], "verification": str(JOB / "verification.json")})
    print(json.dumps(report, ensure_ascii=False), flush=True)


def wait_existing():
    JOB.mkdir(parents=True, exist_ok=True)
    with (JOB / "worker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads((JOB / "progress.json").read_text())
        state["worker_pid"] = os.getpid(); save("progress.json", state)
        while True:
            health = request("/health", timeout=10)
            snapshot = request("/kg/datasets/" + state["dataset"] + "/progress", timeout=30)
            count = 0
            for row in snapshot["data"]:
                slot = (row.get("pipeline_status") or {}).get("custom_pipeline", {}).get(snapshot["dataset_id"])
                if (slot.get("status") if isinstance(slot, dict) else slot) == "DATA_ITEM_PROCESSING_COMPLETED": count += 1
            state.update(completed_papers=count, updated_at=datetime.now(timezone.utc).isoformat())
            save("progress.json", state)
            if health.get("knowledge_operation") != "evidence-index": break
            time.sleep(10)
        verify()


if __name__ == "__main__":
    if "--wait-existing" in sys.argv:
        wait_existing()
    elif "--verify-only" in sys.argv:
        verify()
    else:
        run()
