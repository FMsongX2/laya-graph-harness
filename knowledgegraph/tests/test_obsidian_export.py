"""Read-only Obsidian view: only re-verified approved records, resolvable links, safe regeneration."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("export_obsidian_test", ROOT / "tools/export_obsidian.py")
exporter = importlib.util.module_from_spec(spec); spec.loader.exec_module(exporter)


def approved(paper, number, source, relation, target, quote, conditions=(), **extra):
    """The shape accept_semantic.py writes for a source-reviewed assertion (synthetic content)."""
    return {"id": f"{paper}-A{number:03d}", "source": {"name": source[0], "type": source[1], "aliases": list(source[2:])},
            "relation": relation, "target": {"name": target[0], "type": target[1], "aliases": []},
            "text": f"{source[0]} {relation.replace('_', ' ')} {target[0]} in the reported setting.",
            "conditions": list(conditions),
            "evidence": [{"page": 2, "quote": quote, "exact_quote_match": True,
                          "quote_sha256": hashlib.sha256(quote.encode()).hexdigest(), **extra}],
            "paper_id": paper, "paper_title": f"Synthetic paper {paper}", "original_pdf": "private.pdf",
            "pdf_sha256": "b" * 64, "source_url": "https://example.invalid/" + paper,
            "verification_status": "source_reviewed", "review_reason": "Quote states the relation.",
            "reviewer_action": "supported", "scientific_truth_verified": False}


class ObsidianExport(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); base = Path(self.temporary.name)
        self.records, self.out = base / "records", base / "vault/knowledge"
        self.save(approved("P01", 1, ("Dense retriever", "Method", "DPR"), "uses_dataset", ("Natural Questions", "Dataset"),
                           "We train the dense retriever on Natural Questions.", ["Open-domain QA only"]))
        self.save(approved("P01", 2, ("Dense retriever", "Method"), "has_limitation", ("Lexical mismatch", "Finding"),
                           "Rare entity names remain difficult for the retriever."))
        # Same name with another type, and names Windows cannot use as files.
        self.save(approved("P02", 1, ("CON", "Model"), "uses_metric", ("Recall@20", "Metric"),
                           "Recall@20 is reported for every model.", evidence_type="table_cell",
                           table_id="T2", row_label="CON", column_label="R@20"))
        self.save(approved("P02", 2, ("Dense retriever", "Model"), "alternative_to", ("a/b: c? [x]", "Method"),
                           "The model is compared with the a/b: c? [x] baseline."))

    def tearDown(self): self.temporary.cleanup()

    def save(self, record):
        path = self.records / record["paper_id"] / (record["id"] + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        # The retrieval Markdown written next to each record must be ignored.
        path.with_suffix(".md").write_text("retrieval text", encoding="utf-8")

    def export(self): return exporter.export(self.out, self.records)

    def notes(self): return {p.relative_to(self.out).as_posix(): p.read_text(encoding="utf-8") for p in self.out.rglob("*.md")}

    def test_notes_links_resolve_and_keep_evidence_conditions(self):
        result = self.export()
        self.assertEqual(result["assertions"], 4)
        notes = self.notes()
        for name, text in notes.items():
            self.assertIn('scientific_truth_certified: false',text)
            self.assertIn('not a certification',text)
            for bracketed, plain in re.findall(r"\]\((?:<([^>]+)>|([^)\s]+))\)", text):
                target = bracketed or plain
                if target.startswith("http"): continue
                self.assertTrue((self.out / name).parent.joinpath(target).resolve().is_file(), f"{name} -> {target}")
        note = notes["assertions/P01/P01-A001.md"]
        self.assertIn("> We train the dense retriever on Natural Questions.", note)
        self.assertIn("- Open-domain QA only", note)
        self.assertIn('scientific_truth_certified: false', note)
        self.assertIn("table T2, row CON, column R@20", notes["assertions/P02/P02-A001.md"])
        self.assertNotIn("private.pdf", "".join(notes.values()))
        # Same display name, different types: separate notes, reviewed aliases kept.
        self.assertIn("entities/Dense retriever (Method).md", notes)
        self.assertIn("entities/Dense retriever (Model).md", notes)
        self.assertIn('  - "DPR"', notes["entities/Dense retriever (Method).md"])

    def test_file_names_are_portable(self):
        self.export()
        for path in self.out.rglob("*"):
            self.assertIsNone(re.search(r'[<>:"\\|?*]', path.name), path.name)
            self.assertNotIn(path.stem.upper(), exporter.RESERVED)

    def test_regeneration_is_deterministic_and_removes_stale_notes(self):
        self.export(); first = self.notes()
        self.export(); self.assertEqual(self.notes(), first)
        (self.records / "P02/P02-A002.json").unlink()
        result = self.export()
        self.assertIn("assertions/P02/P02-A002.md", result["removed_stale"])
        self.assertNotIn("assertions/P02/P02-A002.md", self.notes())

    def test_hand_edited_note_and_foreign_folder_are_never_overwritten(self):
        self.export()
        note = self.out / "assertions/P01/P01-A001.md"
        note.write_text(note.read_text(encoding="utf-8") + "\nmy own thought\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "edited"): self.export()
        self.assertIn("my own thought", note.read_text(encoding="utf-8"))
        foreign = self.out.parent / "personal"; foreign.mkdir(); (foreign / "diary.md").write_text("mine")
        with self.assertRaisesRegex(ValueError, "not created by this export"): exporter.export(foreign, self.records)
        self.assertEqual((foreign / "diary.md").read_text(), "mine")

    def test_unverified_or_tampered_record_blocks_the_whole_export(self):
        tampered = json.loads((self.records / "P01/P01-A002.json").read_text(encoding="utf-8"))
        tampered["evidence"][0]["quote"] = "A different sentence than the reviewed one."
        (self.records / "P01/P01-A002.json").write_text(json.dumps(tampered), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "reviewed hash"): self.export()
        self.assertFalse(self.out.exists())
        draft = approved("P03", 1, ("Method X", "Method"), "solves", ("Task Y", "Problem"), "Method X solves task Y here.")
        draft["verification_status"] = "draft"
        tampered["evidence"][0]["quote_sha256"] = hashlib.sha256(tampered["evidence"][0]["quote"].encode()).hexdigest()
        (self.records / "P01/P01-A002.json").write_text(json.dumps(tampered), encoding="utf-8")
        self.save(draft)
        with self.assertRaisesRegex(ValueError, "not a source-reviewed record"): self.export()

    def test_new_generated_name_cannot_overwrite_an_unowned_note(self):
        self.export()
        note=self.out/'entities/Future method.md';note.write_text('my personal note',encoding='utf-8')
        self.save(approved('P03',1,('Future method','Method'),'solves',('New task','Problem'),'The future method solves this task.'))
        with self.assertRaisesRegex(ValueError,'unowned'):self.export()
        self.assertEqual(note.read_text(encoding='utf-8'),'my personal note')

    def test_manifest_paths_cannot_escape_the_export_root(self):
        self.export();victim=self.out.parent/'keep.md';victim.write_text('keep me',encoding='utf-8')
        manifest=self.out/exporter.MANIFEST;data=json.loads(manifest.read_text(encoding='utf-8'))
        data['files']['../keep.md']=exporter.sha(victim.read_bytes())
        manifest.write_text(json.dumps(data),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'path'):self.export()
        self.assertEqual(victim.read_text(encoding='utf-8'),'keep me')

    def test_symlinked_output_subfolder_is_refused_before_writes(self):
        self.out.mkdir(parents=True);outside=self.out.parent/'outside';outside.mkdir()
        try:(self.out/'entities').symlink_to(outside,target_is_directory=True)
        except OSError:self.skipTest('Directory symlinks unavailable on this host')
        (self.out/exporter.MANIFEST).write_text(json.dumps({'format':exporter.FORMAT,'files':{}}),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'path'):self.export()
        self.assertFalse(any(outside.iterdir()))


if __name__ == "__main__":
    unittest.main()
