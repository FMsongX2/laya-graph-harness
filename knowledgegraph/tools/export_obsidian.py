"""Export approved source assertions as a read-only Obsidian/Markdown view.

One-way: the graph and approved records stay the source of truth. Notes are regenerated, never
read back, and the export refuses to overwrite a folder it does not own or notes edited by hand.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))
from semantic_models import SemanticAssertion

RECORDS = ROOT / "data/semantic-assertions"
MANIFEST = ".laya-export.json"
FORMAT = "laya-obsidian-v1"
RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def sha(data):
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def load_records(root):
    """Approved records only, re-checked; any failure refuses the whole export."""
    records, problems = [], []
    for path in sorted(Path(root).glob("P[0-9][0-9]/P[0-9][0-9]-A*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            SemanticAssertion.model_validate(record)
            if record["id"] != path.stem or not re.fullmatch(re.escape(path.parent.name) + r"-A\d+", record["id"]):
                raise ValueError("assertion ID does not match its paper folder and file name")
            if record.get("paper_id") != path.parent.name:
                raise ValueError("paper_id does not match its folder")
            if record.get("verification_status") != "source_reviewed":
                raise ValueError("not a source-reviewed record")
            if not re.fullmatch(r"[0-9a-f]{64}", record.get("pdf_sha256", "")):
                raise ValueError("missing PDF SHA-256")
            for evidence in record["evidence"]:
                if evidence.get("exact_quote_match") is not True or evidence.get("quote_sha256") != sha(evidence["quote"]):
                    raise ValueError("evidence quote differs from its reviewed hash")
            records.append(record)
        except (OSError, ValueError, KeyError, TypeError) as error:
            problems.append(f"{path.relative_to(root)}: {str(error).splitlines()[0]}")
    if problems:
        raise ValueError("Approved records failed re-verification; nothing exported:\n" + "\n".join(problems))
    return records


def filename(name):
    """A readable file name that is valid on Windows, macOS and Linux."""
    clean = re.sub(r'[<>:"/\\|?*\x00-\x1f#^\[\]]', "-", name).strip().rstrip(". ")
    clean = re.sub(r"\s+", " ", clean)[:80].rstrip(". ") or "unnamed"
    return clean + "_" if clean.split(".")[0].upper() in RESERVED else clean


def link(text, source, target):
    """Relative Markdown link (Obsidian 'Use [[Wikilinks]]' off), as PersonaGraph notes use."""
    path = os.path.relpath(target, Path(source).parent).replace(os.sep, "/")
    label = text.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")
    return f"[{label}](<{path}>)" if re.search(r"[\s()]", path) else f"[{label}]({path})"


def frontmatter(values):
    lines = ["---"]
    for key, value in values.items():
        if isinstance(value, list):
            lines.append(f"{key}:" + ("" if value else " []"))
            lines += [f"  - {json.dumps(item, ensure_ascii=False)}" for item in value]
        else:
            lines.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
    return "\n".join(lines + ["---", ""])


def quote_block(text):
    return "\n".join("> " + line for line in text.splitlines() or [""])


def render(records):
    """{relative path: text} for the whole view. Deterministic for identical records."""
    entities = {}
    for record in records:
        for role in ("source", "target"):
            entity = record[role]
            key = (entity["name"].casefold(), entity["type"])
            entry = entities.setdefault(key, {"name": entity["name"], "type": entity["type"], "aliases": set(),
                                              "source": [], "target": []})
            entry["aliases"].update(entity.get("aliases", []))
            entry[role].append(record)
    # Readable unique file names (case-insensitive file systems): add the type where names
    # collide, then a short hash if still taken. Sorted keys keep the choice deterministic.
    names = defaultdict(int)
    for entity in entities.values(): names[filename(entity["name"]).casefold()] += 1
    entity_path, used = {}, set()
    for key in sorted(entities):
        name = filename(entities[key]["name"])
        if names[name.casefold()] > 1: name = filename(f"{entities[key]['name']} ({key[1]})")
        if name.casefold() in used: name += " " + sha(repr(key))[:8]
        used.add(name.casefold())
        entity_path[key] = Path("entities") / (name + ".md")
    papers = defaultdict(list)
    for record in records: papers[record["paper_id"]].append(record)
    assertion_path = {r["id"]: Path("assertions") / r["paper_id"] / (r["id"] + ".md") for r in records}
    paper_path = {p: Path("papers") / (p + ".md") for p in papers}
    files = {}
    caution = ("Source-reviewed assertion: the quotes match the reviewed PDF. This is not a certification "
               "of scientific truth or of performance in your environment. Generated, read-only view.")

    for record in records:
        here = assertion_path[record["id"]]
        source_key = (record["source"]["name"].casefold(), record["source"]["type"])
        target_key = (record["target"]["name"].casefold(), record["target"]["type"])
        body = frontmatter({"id": record["id"], "kind": "laya-assertion", "paper": record["paper_id"],
                            "relation": record["relation"], "source": record["source"]["name"],
                            "target": record["target"]["name"], "verification_status": record["verification_status"],
                            "scientific_truth_certified": False, "pdf_sha256": record["pdf_sha256"],
                            "tags": ["laya/assertion", "laya/relation/" + record["relation"]]})
        body += (f"# {record['source']['name']} — {record['relation'].replace('_', ' ')} — {record['target']['name']}\n\n"
                 f"{link(record['source']['name'], here, entity_path[source_key])} **{record['relation'].replace('_', ' ')}** "
                 f"{link(record['target']['name'], here, entity_path[target_key])}\n\n"
                 f"{record['text']}\n\n## Conditions\n\n")
        body += "\n".join("- " + c for c in record["conditions"]) if record["conditions"] else "Not further specified by the source."
        body += "\n\n## Source evidence\n\n"
        for evidence in record["evidence"]:
            where = f"PDF page {evidence['page']}"
            if evidence.get("table_id"):
                where += f", table {evidence['table_id']}, row {evidence.get('row_label')}, column {evidence.get('column_label')}"
            body += f"{where}:\n\n{quote_block(evidence['quote'])}\n\n"
        body += (f"## Provenance\n\n- Paper: {link(record['paper_id'] + ': ' + record['paper_title'], here, paper_path[record['paper_id']])}\n"
                 f"- Review: {record.get('reviewer_action', 'supported')} — {record.get('review_reason', '')}\n"
                 f"- Source: {record.get('source_url', '')}\n\n> [!note] {caution}\n")
        files[here] = body

    for paper, items in sorted(papers.items()):
        here, first = paper_path[paper], items[0]
        body = frontmatter({"id": paper, "kind": "laya-paper", "title": first["paper_title"],
                            "source_url": first.get("source_url", ""), "pdf_sha256": first["pdf_sha256"],
                            "assertions": len(items), "scientific_truth_certified":False,"tags": ["laya/paper"]})
        body += f"# {paper}: {first['paper_title']}\n\n{first.get('source_url', '')}\n\n## Reviewed assertions\n\n"
        body += "\n".join(f"- {link(r['id'], here, assertion_path[r['id']])}: {r['source']['name']} "
                          f"*{r['relation'].replace('_', ' ')}* {r['target']['name']}" for r in sorted(items, key=lambda r: r["id"]))
        files[here] = body + f"\n\n> [!note] {caution}\n"

    for key, entity in sorted(entities.items()):
        here = entity_path[key]
        body = frontmatter({"kind": "laya-entity", "name": entity["name"], "entity_type": entity["type"],
                            "aliases": sorted(entity["aliases"]), "scientific_truth_certified":False,
                            "tags": ["laya/entity/" + entity["type"]]})
        body += f"# {entity['name']}\n\n*{entity['type']}*"
        if entity["aliases"]: body += " · reviewed aliases: " + ", ".join(sorted(entity["aliases"]))
        for role, heading in (("source", "As source"), ("target", "As target")):
            if entity[role]:
                body += f"\n\n## {heading}\n\n" + "\n".join(
                    f"- {link(r['id'], here, assertion_path[r['id']])}: {r['source']['name']} "
                    f"*{r['relation'].replace('_', ' ')}* {r['target']['name']} ({r['paper_id']})"
                    for r in sorted(entity[role], key=lambda r: r["id"]))
        files[here] = body + f"\n\n> [!note] {caution}\n"

    index = Path("README.md")
    relations = defaultdict(int)
    for record in records: relations[record["relation"]] += 1
    body = frontmatter({"kind": "laya-export-index", "format": FORMAT, "assertions": len(records),
                        "papers": len(papers), "entities": len(entities),
                        "scientific_truth_certified":False,"tags": ["laya/index"]})
    body += ("# Laya reviewed knowledge\n\nGenerated from approved source assertions by `kg export-obsidian`. "
             "Do not edit these notes: they are replaced on the next export, and an edited note blocks it. "
             "Link to them from your own notes instead.\n\n## Papers\n\n")
    body += "\n".join(f"- {link(p + ': ' + items[0]['paper_title'], index, paper_path[p])} ({len(items)})"
                      for p, items in sorted(papers.items()))
    body += "\n\n## Relations\n\n" + "\n".join(f"- {name.replace('_', ' ')}: {count}" for name, count in sorted(relations.items()))
    files[index] = body + f"\n\n> [!note] {caution}\n"
    return {path.as_posix(): text for path, text in files.items()}


def safe_target(out, name):
    """Manifest and generated paths must stay within the owned, unsymlinked tree."""
    if not isinstance(name,str) or not name or '\\' in name or ':' in name:
        raise ValueError('Invalid export path')
    relative=PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or relative.as_posix()!=name or name==MANIFEST:
        raise ValueError('Invalid export path')
    target=out
    for part in relative.parts:
        target=target/part
        if target.is_symlink():raise ValueError('Symlinked export path is not owned')
    if not target.resolve().is_relative_to(out):raise ValueError('Export path leaves its root')
    return target


def write(out, files):
    """Replace the previous export it owns; refuse foreign folders and hand-edited notes."""
    out = Path(out).resolve()
    manifest_path = out / MANIFEST
    if out.exists() and any(out.iterdir()) and not manifest_path.exists():
        raise ValueError(f"{out} is not empty and was not created by this export; choose an empty or previous export folder")
    if manifest_path.is_symlink():raise ValueError('Symlinked manifest path is not owned')
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"format":FORMAT,"files": {}}
    if not isinstance(previous,dict) or previous.get("format") != FORMAT or not isinstance(previous.get('files'),dict):
        raise ValueError("Unknown export format in " + str(manifest_path))
    old_targets={name:safe_target(out,name) for name in previous['files']}
    targets={name:safe_target(out,name) for name in files}
    if any(not isinstance(digest,str) or not re.fullmatch(r'[0-9a-f]{64}',digest) for digest in previous['files'].values()):
        raise ValueError('Invalid exported file hash')
    unowned=[name for name,target in targets.items() if target.exists() and name not in previous['files']]
    if unowned:raise ValueError('Generated names would overwrite unowned notes: '+', '.join(unowned))
    edited = [name for name, digest in previous["files"].items()
              if old_targets[name].exists() and sha(old_targets[name].read_bytes()) != digest]
    if edited:
        raise ValueError("Exported notes were edited; move your changes to your own notes, then delete these files:\n"
                         + "\n".join(edited))
    for name in previous["files"]:
        if name not in files and old_targets[name].exists(): old_targets[name].unlink()
    digests = {}
    for name, text in sorted(files.items()):
        target = targets[name]
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = text.encode("utf-8")
        fd,tempname=tempfile.mkstemp(prefix='.laya-write-',dir=target.parent)
        temporary=Path(tempname)
        try:
            with os.fdopen(fd,'wb') as stream:stream.write(raw)
            temporary.replace(target)
        finally:
            if temporary.exists():temporary.unlink()
        digests[name] = sha(raw)
    for folder in sorted({(out / name).parent for name in previous["files"]}, key=lambda p: len(p.parts), reverse=True):
        if folder != out and folder.is_dir() and not any(folder.iterdir()): folder.rmdir()
    manifest = {"format": FORMAT, "generated_at": datetime.now(timezone.utc).isoformat(), "files": digests}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"out": str(out), "notes": len(files), "removed_stale": sorted(set(previous["files"]) - set(files)),
            "read_only_view": True, "source_of_truth": "approved records and graph"}


def export(out, records_root=RECORDS):
    records = load_records(records_root)
    if not records: raise ValueError(f"No approved source records under {records_root}")
    result = write(out, render(records))
    return {**result, "assertions": len(records)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="empty folder or a previous export, e.g. <vault>/knowledge")
    print(json.dumps(export(parser.parse_args().out), ensure_ascii=False, indent=2))
