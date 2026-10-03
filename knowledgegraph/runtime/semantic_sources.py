"""Attach approved source coordinates only to the matching indexed text version."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def indexed_text_matches(text, assertion):
    expected = assertion.get("retrieval_text")
    return isinstance(text, str) and isinstance(expected, str) and text.strip() == expected.strip()


def attach_semantic_sources(result, root=None):
    registry_path = (root or ROOT) / "state/semantic-ingestion/assertions.json"
    if not registry_path.exists():
        return result
    records = {Path(item["markdown_path"]).stem: item
               for item in json.loads(registry_path.read_text())["assertions"]}
    def walk(value):
        if isinstance(value, dict):
            record = records.get(value.get("document_name"))
            if record:
                assertion = json.loads(Path(record["json_path"]).read_text())
                if indexed_text_matches(value.get("text"), assertion):
                    value["semantic_assertion"] = assertion
                else:
                    value.pop("semantic_assertion", None)
                    value["semantic_source_status"] = {
                        "status": "indexed_text_does_not_match_current_approved_record",
                        "assertion_id": assertion["id"],
                        "source_review_applies_to_this_text": False}
            for child in list(value.values()):
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(result)
    return result
