"""Publish accepted source-reviewed relationships into the shared Cognee graph."""
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from cli import request


def publish(papers):
    records = json.loads((ROOT / "state/semantic-ingestion/assertions.json").read_text())["assertions"]
    records = [item for item in records if item["paper_id"] in papers]
    if not records: raise ValueError("No accepted assertions for the requested papers")
    dataset = json.loads((ROOT / "config/settings.json").read_text())["knowledge"]["semantic_dataset"]
    for paper in papers:
        selected = [item for item in records if item["paper_id"] == paper]
        files = [item["markdown_path"] for item in selected]
        added = request("/kg/add", {"files": files, "dataset": dataset, "domains": ["source-reviewed", paper]}, timeout=3600)
        rows = request("/kg/datasets/" + dataset + "/data")["data"]
        names = {Path(file).stem for file in files}
        ids = [row["id"] for row in rows if row["name"] in names]
        assert len(ids) == len(files)
        result = request("/kg/semantic-index", {"data_ids": ids}, timeout=7200)
        out = ROOT / "state/semantic-ingestion/results";out.mkdir(exist_ok=True)
        (out / (paper + "-index.json")).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"paper": paper, "indexed_assertions": len(ids)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    publish(sys.argv[1:])
