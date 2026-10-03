"""Accept only independently reviewed, source-quote-matched semantic assertions."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import re
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))
from semantic_models import SemanticAssertion
JOB = ROOT / "state/semantic-ingestion"


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def accept(paper):
    spec = json.loads((JOB / "sources" / (paper + ".json")).read_text())
    draft_path = JOB / "drafts" / (paper + ".json")
    review_path = JOB / "reviews" / (paper + ".json")
    draft = json.loads(draft_path.read_text());review = json.loads(review_path.read_text())
    expected_pages = set(range(1, spec["pages"] + 1))
    assert set(draft["read_pages"]) == expected_pages, "Author source coverage is incomplete"
    assert set(review["read_pages"]) == expected_pages, "Reviewer source coverage is incomplete"
    pages = {item["page"]: item["text"] for item in
             (json.loads(line) for line in Path(spec["source_pages_path"]).read_text().splitlines())}
    verdicts = {item["assertion_id"]: item for item in review["reviews"]}
    ids = [item["id"] for item in draft["assertions"]]
    assert len(set(ids)) == len(ids) and len(verdicts) == len(review["reviews"])
    assert set(ids) == set(verdicts), "Every assertion requires one independent review"
    for item in review["reviews"]:
        if item.get("verdict") not in {"supported", "amend", "reject"}:
            raise ValueError("Unknown review verdict: " + str(item.get("verdict")))
        if not isinstance(item.get("reason"), str) or not item["reason"].strip():
            raise ValueError("Every review needs an explicit source-grounded reason")
        if item["verdict"] == "amend" and not item.get("corrected_assertion"):
            raise ValueError("An amendment requires its reviewed corrected assertion")
    aliases = {(item["assertion_id"], item.get("entity_role"), item["alias"]): item
               for item in review.get("alias_reviews", [])}
    if len(aliases) != len(review.get("alias_reviews", [])):
        raise ValueError("Duplicate alias reviews cannot be resolved by overwrite")
    author_notes = draft.get("unresolved", [])
    resolved_notes = review.get("resolved_author_notes", [])
    for item in resolved_notes:
        if item.get("author_note") not in author_notes:
            raise ValueError("A resolved author note must identify an original note exactly")
        if not isinstance(item.get("resolution"), str) or not item["resolution"].strip():
            raise ValueError("Resolved author notes require an explicit reviewed correction")
    accepted = [];rejected = []; alias_unresolved = []
    for original in draft["assertions"]:
        verdict = verdicts[original["id"]]
        if verdict["verdict"] == "reject":
            rejected.append({"assertion": original, "reason": verdict["reason"]});continue
        # A supported review may enrich source coordinates without changing the
        # claim. Prefer that explicitly reviewed canonical record when supplied.
        candidate = verdict.get("corrected_assertion") or original
        try:
            validated = SemanticAssertion.model_validate(candidate).model_dump()
            assert validated["id"] == original["id"]
            assert re.fullmatch(re.escape(paper) + r"-A\d+", validated["id"]), "Assertion ID belongs to another source paper"
            for evidence in validated["evidence"]:
                assert evidence["page"] in pages
                assert normalize(evidence["quote"]) in normalize(pages[evidence["page"]]), "Quote not found on its source PDF page"
                if len(evidence["quote"]) < 8:
                    assert evidence.get("evidence_type") == "table_cell", "Short evidence requires reviewed table-cell context"
                    assert all(evidence.get(key) for key in ["table_id", "row_label", "column_label"]), "Missing table row/column coordinates"
                evidence["exact_quote_match"] = True
                evidence["quote_sha256"] = hashlib.sha256(evidence["quote"].encode()).hexdigest()
            for role in ["source", "target"]:
                kept = []
                for alias in validated[role].get("aliases", []):
                    audit = aliases.get((validated["id"], role, alias))
                    if audit and audit["verdict"] == "supported": kept.append(alias)
                    else: alias_unresolved.append({"assertion_id": validated["id"], "role": role, "alias": alias})
                validated[role]["aliases"] = kept
            validated.update(paper_id=paper, paper_title=spec["title"], original_pdf=spec["pdf_path"],
                pdf_sha256=spec["pdf_sha256"], source_url=spec["source_url"],
                verification_status="source_reviewed", review_reason=verdict["reason"],
                reviewer_action=verdict["verdict"], scientific_truth_verified=False,
                source_draft_sha256=hashlib.sha256(draft_path.read_bytes()).hexdigest(),
                source_review_sha256=hashlib.sha256(review_path.read_bytes()).hexdigest())
            accepted.append(validated)
        except (ValueError, AssertionError, TypeError) as error:
            rejected.append({"assertion": candidate, "reason": str(error), "mechanical_gate_failed": True})
    registry_path = JOB / "assertions.json"
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {"assertions": []}
    registry["assertions"] = [item for item in registry["assertions"] if item["paper_id"] != paper]
    records = []
    for assertion in accepted:
        folder = ROOT / "data/semantic-assertions" / paper;folder.mkdir(parents=True, exist_ok=True)
        path = folder / (assertion["id"] + ".json"); markdown = folder / (assertion["id"] + ".md")
        text = f"# {assertion['source']['name']} {assertion['relation']} {assertion['target']['name']}\n\n"
        text += assertion["text"] + "\n\n## Scope and conditions\n"
        text += "\n".join("- " + condition for condition in assertion["conditions"]) or "Not further specified by this assertion."
        text += "\n\n## Source evidence\n"
        for evidence in assertion["evidence"]:
            text += f"PDF page {evidence['page']}: {evidence['quote']}\n\n"
        text += f"Source: {spec['title']}\n{spec['source_url']}\nVerification: source-reviewed; scientific truth and our-environment performance not certified.\n"
        assertion["retrieval_text"] = text;save(path, assertion);markdown.write_text(text)
        records.append({"assertion_id": assertion["id"], "paper_id": paper, "json_path": str(path), "markdown_path": str(markdown)})
    registry["assertions"] += records;save(registry_path, registry)
    unresolved_author = [note for note in author_notes if not any(item["author_note"] == note for item in resolved_notes)]
    report = {"paper_id": paper, "accepted": len(accepted), "rejected": len(rejected), "rejections": rejected,
              "unresolved": unresolved_author + review.get("unresolved", []),
              "author_unresolved_notes": author_notes, "reviewer_unresolved_notes": review.get("unresolved", []),
              "resolved_author_notes": resolved_notes, "alias_unresolved": alias_unresolved,
              "checked_at": datetime.now(timezone.utc).isoformat(), "source_coverage": spec["pages"]}
    save(JOB / "accepted" / (paper + ".json"), report)
    print(json.dumps({"paper": paper, "accepted": len(accepted), "rejected": len(rejected), "alias_unresolved": len(alias_unresolved)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument("papers", nargs="+")
    for paper in parser.parse_args().papers:accept(paper)
