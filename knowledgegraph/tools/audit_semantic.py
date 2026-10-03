"""Check approved source artifacts without opening graph/vector stores or calling models."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))
from semantic_models import SemanticAssertion


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def reviewed_content(assertion):
    """Compare meaning-bearing fields, excluding gate-added audit metadata."""
    value = SemanticAssertion.model_validate(assertion).model_dump()
    result = {key: value[key] for key in ["id", "relation", "text", "conditions"]}
    for role in ["source", "target"]:
        result[role] = {key: value[role][key] for key in ["name", "type"]}
    result["evidence"] = [{key: item for key, item in quote.items()
                           if key not in {"exact_quote_match", "quote_sha256"}} for quote in value["evidence"]]
    return result


def audit(root=ROOT, all_approved=False, verify_pdfs=False):
    job = root / "state/semantic-ingestion"
    progress = json.loads((job / "progress.json").read_text())
    registry = json.loads((job / "assertions.json").read_text())["assertions"]
    published = set(progress["completed_papers"])
    partial = {paper for paper, state in progress["paper_status"].items()
               if state.get("status") == "published-with-pending-corrections"}
    selected = [record for record in registry if all_approved or record["paper_id"] in published | partial]
    problems = []; sources = {}; counts = {}; identities = set()
    for record in selected:
        identity = (record["paper_id"], record["assertion_id"])
        if identity in identities:
            problems.append({"record": identity, "failed": "duplicate_registry_identity"})
        identities.add(identity)
        try:
            assertion = json.loads(Path(record["json_path"]).read_text())
            SemanticAssertion.model_validate(assertion)
            paper = assertion["paper_id"]
            if paper not in sources:
                spec = json.loads((job / "sources" / (paper + ".json")).read_text())
                pages = {item["page"]: item["text"] for item in
                         (json.loads(line) for line in Path(spec["source_pages_path"]).read_text().splitlines())}
                draft_path, review_path = job / "drafts" / (paper + ".json"), job / "reviews" / (paper + ".json")
                draft = json.loads(draft_path.read_text()); review = json.loads(review_path.read_text())
                sources[paper] = {"spec": spec, "pages": pages, "draft": draft, "review": review,
                                  "draft_hash": digest(draft_path), "review_hash": digest(review_path)}
                expected = set(range(1, spec["pages"] + 1))
                if set(pages) != expected or set(draft["read_pages"]) != expected or set(review["read_pages"]) != expected:
                    problems.append({"paper": paper, "failed": "original_page_or_recorded_read_coverage"})
                if verify_pdfs and digest(spec["pdf_path"]) != spec["pdf_sha256"]:
                    problems.append({"paper": paper, "failed": "original_pdf_hash"})
            source = sources[paper]
            reviews = [item for item in source["review"]["reviews"] if item["assertion_id"] == assertion["id"]]
            draft_assertions = [item for item in source["draft"]["assertions"] if item["id"] == assertion["id"]]
            canonical = (reviews[0].get("corrected_assertion") or draft_assertions[0]) if len(reviews) == 1 and len(draft_assertions) == 1 else None
            checks = {
                "registry_identity": identity == (paper, assertion["id"]),
                "approved_markdown_matches": Path(record["markdown_path"]).read_text() == assertion["retrieval_text"],
                "draft_hash_matches": source["draft_hash"] == assertion["source_draft_sha256"],
                "review_hash_matches": source["review_hash"] == assertion["source_review_sha256"],
                "source_pdf_hash_reference": source["spec"]["pdf_sha256"] == assertion["pdf_sha256"],
                "independent_accepted_review": len(reviews) == 1 and reviews[0]["verdict"] in {"supported", "amend"},
                "approved_content_matches_review": canonical is not None
                    and reviewed_content(assertion) == reviewed_content(canonical),
                "source_reviewed_without_truth_certification": assertion["verification_status"] == "source_reviewed"
                    and assertion["scientific_truth_verified"] is False,
                "source_quotes": bool(assertion["evidence"]) and all(
                    normalize(item["quote"]) in normalize(source["pages"][item["page"]])
                    for item in assertion["evidence"]),
                "quote_verification_metadata": all(item.get("exact_quote_match") is True
                    and item.get("quote_sha256") == hashlib.sha256(item["quote"].encode()).hexdigest()
                    for item in assertion["evidence"]),
            }
            for check, okay in checks.items():
                if not okay:
                    problems.append({"assertion": assertion["id"], "failed": check})
            for role in ["source", "target"]:
                for alias in assertion[role].get("aliases", []):
                    matches = [item for item in source["review"].get("alias_reviews", [])
                               if (item["assertion_id"], item.get("entity_role"), item["alias"]) ==
                                  (assertion["id"], role, alias)]
                    if len(matches) != 1 or matches[0]["verdict"] != "supported":
                        problems.append({"assertion": assertion["id"], "failed": "alias_source_review", "alias": alias})
            counts[paper] = counts.get(paper, 0) + 1
        except (OSError, ValueError, KeyError, TypeError) as error:
            problems.append({"record": identity, "failed": "artifact_read_or_schema", "error": str(error)})
    for paper in published | partial:
        state = progress["paper_status"][paper]
        expected_count = state.get("published_assertions", state["accepted_assertions"])
        found = counts.get(paper, 0)
        if (paper in published and found != expected_count) or (paper in partial and found < expected_count):
            problems.append({"paper": paper, "failed": "published_registry_count", "expected": expected_count,
                             "found": counts.get(paper, 0)})
    if sum(progress["paper_status"][paper].get("published_assertions", progress["paper_status"][paper]["accepted_assertions"])
           for paper in published | partial) != progress["indexed_assertions"]:
        problems.append({"failed": "published_progress_count"})
    return {"checked_at": datetime.now(timezone.utc).isoformat(), "passed": not problems,
            "scope": "all-approved" if all_approved else "published", "records_checked": len(selected),
            "paper_counts": counts, "published_papers": sorted(published),
            "partially_published_papers": sorted(partial),
            "indexed_assertions_reported": progress["indexed_assertions"], "original_pdf_hashes_checked": verify_pdfs,
            "issues": problems,
            "limitations": "Artifact consistency, quote presence and recorded source review; not an independent entailment, scientific-truth or retrieval-recall evaluation."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-approved", action="store_true")
    parser.add_argument("--verify-pdfs", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(all_approved=args.all_approved, verify_pdfs=args.verify_pdfs)
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(encoded)
        temporary.replace(args.output)
    print(encoded, end="")
    raise SystemExit(0 if result["passed"] else 1)
