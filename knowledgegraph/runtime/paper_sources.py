"""Attach original-paper references to retrieved chunks without inventing page coordinates."""
import bisect
from functools import lru_cache
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=64)
def source_text(path: str, modified_ns: int):
    source = Path(path)
    text = source.read_text()
    tokens = list(re.finditer(r"\S+", text))
    starts = []
    original_starts = []
    original_ends = []
    cursor = 0
    for token in tokens:
        starts.append(cursor)
        original_starts.append(token.start())
        original_ends.append(token.end())
        cursor += len(token.group()) + 1
    normalized = " ".join(token.group() for token in tokens)
    metadata = json.loads(source.with_suffix(".source.json").read_text())
    return normalized, starts, original_starts, original_ends, metadata


def chunk_source(payload, papers):
    name = Path(payload.get("document_name") or "").stem
    paper = papers.get(name)
    if not paper:
        return None
    path = Path(paper["input_path"])
    normalized, starts, old_starts, old_ends, metadata = source_text(str(path), path.stat().st_mtime_ns)
    needle = " ".join((payload.get("text") or "").split())
    position = normalized.find(needle) if needle else -1
    unique = position >= 0 and normalized.find(needle, position + 1) < 0
    pages = []
    if unique:
        first = bisect.bisect_right(starts, position) - 1
        last = bisect.bisect_right(starts, position + len(needle) - 1) - 1
        original_start = old_starts[first]
        original_end = old_ends[last]
        pages = [span["pdf_page"] for span in metadata["page_spans"]
                 if span["start_char"] < original_end and span["end_char"] > original_start]
    return {"paper_id": paper["paper_id"], "title": paper["title"],
            "original_pdf": metadata["original_pdf"], "pdf_sha256": metadata["pdf_sha256"],
            "source_url": metadata["source_url"], "versioned_pdf_url": metadata["versioned_pdf_url"],
            "prepared_text": str(path), "pdf_pages": pages,
            "chunk_located_uniquely_in_source": unique}


def attach_paper_sources(result):
    manifest_path = ROOT / "state/paper-ingestion/manifest.json"
    if not manifest_path.exists():
        return result
    manifest = json.loads(manifest_path.read_text())
    papers = {Path(paper["input_path"]).stem: paper for paper in manifest["papers"]}
    def walk(value):
        if isinstance(value, dict):
            if "document_name" in value and "text" in value:
                reference = chunk_source(value, papers)
                if reference:
                    value["paper_source"] = reference
            for child in list(value.values()):
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(result)
    return result
