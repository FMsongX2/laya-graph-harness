"""Exercise publication gates and lossless CLI projections without database writes."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = load("accept_semantic_test", ROOT / "tools/accept_semantic.py")
cli = load("cli_test", ROOT / "tools/cli.py")
semantic_sources = load("semantic_sources_test", ROOT / "runtime/semantic_sources.py")
audit_module = load("audit_semantic_test", ROOT / "tools/audit_semantic.py")


class PublicationGate(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.old_root, self.old_job = gate.ROOT, gate.JOB
        gate.ROOT, gate.JOB = self.root, self.root / "state/semantic-ingestion"
        for name in ["sources", "drafts", "reviews"]:
            (gate.JOB / name).mkdir(parents=True)
        pages = self.root / "pages.jsonl"
        pages.write_text(json.dumps({"page": 1, "text": "Method Alpha solves task beta only with labeled input."}))
        source = {"paper_id": "P99", "pages": 1, "title": "Test source", "pdf_path": "test.pdf",
                  "pdf_sha256": "a" * 64, "source_url": "https://example.invalid/source", "source_pages_path": str(pages)}
        (gate.JOB / "sources/P99.json").write_text(json.dumps(source))
        self.assertion = {"id": "P99-A001", "source": {"name": "Method Alpha", "type": "Method"},
                          "relation": "solves", "target": {"name": "Task beta", "type": "Problem"},
                          "text": "Method Alpha solves task beta only with labeled input.",
                          "conditions": ["Labeled input required"],
                          "evidence": [{"page": 1, "quote": "Method Alpha solves task beta only with labeled input."}]}
        self.draft = {"read_pages": [1], "assertions": [self.assertion]}
        self.review = {"read_pages": [1], "reviews": [{"assertion_id": "P99-A001", "verdict": "supported",
                       "reason": "The source explicitly limits this claim to labeled input."}], "alias_reviews": []}

    def tearDown(self):
        gate.ROOT, gate.JOB = self.old_root, self.old_job
        self.temporary.cleanup()

    def run_gate(self):
        (gate.JOB / "drafts/P99.json").write_text(json.dumps(self.draft))
        (gate.JOB / "reviews/P99.json").write_text(json.dumps(self.review))
        gate.accept("P99")

    def test_valid_scope_and_provenance_preserved(self):
        self.run_gate()
        record = json.loads((self.root / "data/semantic-assertions/P99/P99-A001.json").read_text())
        self.assertEqual(record["conditions"], self.assertion["conditions"])
        self.assertTrue(record["evidence"][0]["exact_quote_match"])
        self.assertFalse(record["scientific_truth_verified"])

    def test_unknown_verdict_cannot_publish(self):
        self.review["reviews"][0]["verdict"] = "uncertain"
        with self.assertRaises(ValueError): self.run_gate()
        self.assertFalse((gate.JOB / "assertions.json").exists())

    def test_amendment_requires_reviewed_replacement(self):
        self.review["reviews"][0]["verdict"] = "amend"
        with self.assertRaises(ValueError): self.run_gate()

    def test_wrong_quote_is_rejected(self):
        self.assertion["evidence"][0]["quote"] = "Method Alpha works with any input."
        self.run_gate()
        report = json.loads((gate.JOB / "accepted/P99.json").read_text())
        self.assertEqual((report["accepted"], report["rejected"]), (0, 1))

    def test_unreviewed_alias_never_enters_graph(self):
        self.assertion["source"]["aliases"] = ["Unrelated algorithm"]
        self.run_gate()
        record = json.loads((self.root / "data/semantic-assertions/P99/P99-A001.json").read_text())
        self.assertEqual(record["source"]["aliases"], [])

    def test_duplicate_alias_reviews_cannot_overwrite(self):
        alias = {"assertion_id": "P99-A001", "entity_role": "source", "alias": "Alpha", "verdict": "supported"}
        self.review["alias_reviews"] = [alias, copy.deepcopy(alias)]
        with self.assertRaises(ValueError): self.run_gate()

    def test_only_explicit_review_correction_resolves_author_note(self):
        self.draft["unresolved"] = ["Figure appears to disagree with table", "An assumption is unclear"]
        self.review["resolved_author_notes"] = [{"author_note": self.draft["unresolved"][0],
                     "resolution": "Original figure and table agree; the author's reading was mistaken."}]
        self.run_gate()
        report = json.loads((gate.JOB / "accepted/P99.json").read_text())
        self.assertEqual(report["unresolved"], ["An assumption is unclear"])
        self.assertEqual(report["author_unresolved_notes"], self.draft["unresolved"])
        self.assertEqual(report["resolved_author_notes"], self.review["resolved_author_notes"])

    def test_resolution_cannot_silently_remove_unmatched_note(self):
        self.draft["unresolved"] = ["An assumption is unclear"]
        self.review["resolved_author_notes"] = [{"author_note": "Another note", "resolution": "It is resolved."}]
        with self.assertRaises(ValueError): self.run_gate()

    def write_published_progress(self):
        (gate.JOB / "progress.json").write_text(json.dumps({"completed_papers": ["P99"], "indexed_assertions": 1,
             "paper_status": {"P99": {"accepted_assertions": 1}}}))

    def test_corpus_audit_validates_accepted_review_content(self):
        self.run_gate(); self.write_published_progress()
        report = audit_module.audit(root=self.root)
        self.assertTrue(report["passed"], report["issues"])

    def test_matching_json_and_markdown_cannot_hide_unreviewed_claim_change(self):
        self.run_gate(); self.write_published_progress()
        path = self.root / "data/semantic-assertions/P99/P99-A001.json"
        record = json.loads(path.read_text())
        replacement = "Method Alpha solves task beta without any labeled input."
        record["retrieval_text"] = record["retrieval_text"].replace(record["text"], replacement, 1)
        record["text"] = replacement
        path.write_text(json.dumps(record))
        path.with_suffix(".md").write_text(record["retrieval_text"])
        report = audit_module.audit(root=self.root)
        self.assertFalse(report["passed"])
        self.assertIn("approved_content_matches_review", [item["failed"] for item in report["issues"]])

    def test_partial_native_publication_is_counted_without_claiming_paper_complete(self):
        self.run_gate()
        (gate.JOB / "progress.json").write_text(json.dumps({"completed_papers": [], "indexed_assertions": 1,
             "paper_status": {"P99": {"status": "published-with-pending-corrections", "accepted_assertions": 1,
                                       "published_assertions": 1}}}))
        report = audit_module.audit(root=self.root)
        self.assertTrue(report["passed"], report["issues"])
        self.assertEqual(report["published_papers"], [])
        self.assertEqual(report["partially_published_papers"], ["P99"])


class CompactRetrieval(unittest.TestCase):
    def test_native_semantic_results_keep_every_claim_scope_and_quote(self):
        # Synthetic projection fixture: no local paper corpus is distributed.
        assertion = {"id": "P99-A001", "pdf_sha256": "a" * 64,
                     "text": "Method Alpha requires labeled input for Task Beta.",
                     "conditions": ["Labeled input", "Reported experimental scope"],
                     "evidence": [{"page": 2, "quote": "Method Alpha requires labeled input for Task Beta."}],
                     "verification_status": "source_reviewed", "scientific_truth_verified": False,
                     "retrieval_text": "Repeated retrieval representation " * 80}
        second = {**copy.deepcopy(assertion), "id": "P99-A002", "conditions": ["Different reported scope"]}
        result = {"query": "synthetic test query", "answer_generated": False,
                  "results": [{"objects_result": [{"semantic_assertion": assertion},
                                                   {"nested": [{"semantic_assertion": second}]}],
                               "response_prompt": "Synthetic unused prompt " * 100}]}
        original = []
        def visit(value):
            if isinstance(value, dict):
                if "semantic_assertion" in value: original.append(value["semantic_assertion"])
                else:
                    for item in value.values(): visit(item)
            elif isinstance(value, list):
                for item in value: visit(item)
        visit(result)
        self.assertTrue(original)
        projected = cli.compact_search(result)
        records = {item["id"]: item for item in projected["matches"]}
        for item in original:
            expected = {key: value for key, value in item.items() if key != "retrieval_text"}
            actual = {key: value for key, value in records[item["id"]].items() if key != "kind"}
            self.assertEqual(actual, expected)
        self.assertLess(len(json.dumps(projected)), len(json.dumps(result)))

    def test_native_chunk_list_keeps_original_text_and_source(self):
        result = {"results": [{"objects_result": [
            {"id": "synthetic-chunk", "document_name": "Synthetic source", "chunk_index": 0,
             "text": "Source evidence remains verbatim with its reported condition.",
             "paper_source": {"paper_id": "P99", "pages": [3], "url": "https://example.invalid/source"}}]}]}
        projected = cli.compact_search(result)
        self.assertTrue(projected["matches"])
        for match in projected["matches"]:
            self.assertEqual(match["kind"], "original_text")
            self.assertTrue(match["text"])
            self.assertTrue(match["paper_source"])

    def test_graph_only_context_does_not_invent_source_coordinates(self):
        result = {"results": [{"objects_result": [], "context_result": "An unqualified graph path"}]}
        projected = cli.compact_search(result)
        self.assertEqual(projected["matches"], [])
        self.assertFalse(projected["unstructured_contexts"][0]["source_coordinates_attached"])


class IndexedSourceVersion(unittest.TestCase):
    def test_reused_filename_cannot_certify_an_old_indexed_claim(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            registry = root / "state/semantic-ingestion/assertions.json"
            registry.parent.mkdir(parents=True)
            approved = root / "approved.json"
            approved.write_text(json.dumps({"id": "P99-A001", "retrieval_text": "New reviewed text"}))
            registry.write_text(json.dumps({"assertions": [{"markdown_path": "P99-A001.md", "json_path": str(approved)}]}))
            current = {"document_name": "P99-A001", "text": "New reviewed text\n"}
            stale = {"document_name": "P99-A001", "text": "Old different claim"}
            semantic_sources.attach_semantic_sources([current, stale], root=root)
            self.assertIn("semantic_assertion", current)
            self.assertNotIn("semantic_assertion", stale)
            self.assertFalse(stale["semantic_source_status"]["source_review_applies_to_this_text"])
            projection = cli.compact_search({"results": [{"objects_result": [stale]}]})
            self.assertEqual(projection["matches"][0]["kind"], "unverified_indexed_text")
            self.assertEqual(projection["matches"][0]["text"], stale["text"])


class SourceEvidencePrecheck(unittest.TestCase):
    def test_author_schema_validation_catches_short_algorithm_fragment(self):
        with self.assertRaises(ValueError):
            gate.SemanticAssertion.model_validate({"id": "P99-A001",
                "source": {"name": "Algorithm", "type": "Method"}, "relation": "works_by",
                "target": {"name": "Initialization", "type": "Method"},
                "text": "Algorithm initializes a counter before iteration.",
                "evidence": [{"page": 1, "quote": "D ← 0"}]})

    def test_real_short_table_value_keeps_its_coordinates(self):
        record = gate.SemanticAssertion.model_validate({"id": "P99-A001",
            "source": {"name": "Algorithm", "type": "Method"}, "relation": "reports_result",
            "target": {"name": "A reported score", "type": "Finding"},
            "text": "Algorithm has a reported score in a specified table cell.",
            "evidence": [{"page": 1, "quote": "0.91", "evidence_type": "table_cell",
                          "table_id": "Table 1", "row_label": "Algorithm", "column_label": "Score"}]})
        self.assertEqual(record.evidence[0].model_dump()["row_label"], "Algorithm")

    def test_whitespace_cannot_serve_as_source_evidence(self):
        with self.assertRaises(ValueError):
            gate.SemanticAssertion.model_validate({"id": "P99-A001",
                "source": {"name": "Algorithm", "type": "Method"}, "relation": "works_by",
                "target": {"name": "Initialization", "type": "Method"},
                "text": "Algorithm initializes a counter before iteration.",
                "evidence": [{"page": 1, "quote": " " * 12}]})


if __name__ == "__main__":
    unittest.main()
