# Obsidian view

Obsidian is a reading and linking surface here, not a store. Approved records and the graph stay the source of truth. Agents keep using `kg select`/`kg search`, which verify evidence; nothing reads notes back into the graph.

## Export

```sh
knowledgegraph/kg export-obsidian --out <vault>/knowledge
```

- Every approved record under `data/semantic-assertions` is re-checked first: schema, `source_reviewed` status, and each quote against its reviewed SHA-256. One failure refuses the whole export.
- Output: one note per assertion (statement, conditions, quoted evidence with PDF page/table coordinates, review provenance), per paper and per entity (reviewed aliases become Obsidian aliases), plus `README.md`.
- Notes carry `scientific_truth_certified: false` and a callout: source review is not truth or performance certification.
- Re-running replaces the previous export and removes notes whose records are gone. It refuses to write into a non-empty folder it did not create, and refuses while an exported note has been edited. Write your own thoughts in your own notes and link to exported ones.
- Manifest paths must stay inside the export root. Symlinked output subpaths and new generated names that would overwrite unowned notes are refused before writes; temporary files use unique names.
- Local PDF paths are not written; the export contains quotes, so keep the vault as private as the corpus.

## One vault with PersonaGraph

[PersonaGraph](https://github.com/FMsongX2/personagraph) keeps user-alignment notes as plain Markdown with frontmatter in its `memory/` folder. A single vault can show both, so a user's decision can link to the evidence behind it:

```text
<vault>/
  persona/     -> personagraph/memory   (link, edited through PersonaGraph's review flow)
  knowledge/   -> kg export-obsidian     (generated, read-only)
```

```sh
ln -s /path/to/personagraph/memory <vault>/persona                                  # macOS / Linux
```

```powershell
New-Item -ItemType Junction -Path <vault>\persona -Target C:\path\to\personagraph\memory   # Windows, no admin needed
```

A decision note then cites evidence with an ordinary relative link, e.g. `[P03-A012](../../knowledge/assertions/P03/P03-A012.md)` from `persona/decisions/`.

Recommended vault settings:

- **Files and links → Use [[Wikilinks]]: off**, **New link format: relative path.** Both projects use relative Markdown links, which agents and other tools read too.
- Keep note-rewriting plugins (linters, auto-updated dates, AI note generators) off for `persona/` and `knowledge/`. PersonaGraph records the hash of each reviewed note, and exported notes must stay byte-identical to be regenerated.
- Never place PersonaGraph's `data/` (original evidence, backup password) or this project's `data/`/`state/` inside the vault or a synced folder.
