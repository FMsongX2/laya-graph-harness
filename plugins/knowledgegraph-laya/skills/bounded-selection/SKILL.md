---
name: bounded-selection
description: Use a configured local Cognee/Neo4j/Laya runtime to align a short English paper excerpt with reviewed relationships and collect conditions and evidence in one code call. General problem discovery uses the existing graph search.
allowed-tools: Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/bootstrap.py)
---

# Bounded local graph selection

Preparation: !`python3 "${CLAUDE_SKILL_DIR}/scripts/bootstrap.py"`

Set `LAYA_GRAPH_KG_ROOT` to the single installed KnowledgeGraph project root.
Claude Code preprocesses the command before reading this skill. In other
hosts, use the bundled hook source or run `"$LAYA_GRAPH_KG_ROOT/kg" prepare`
once before querying. Do not start parallel project roots over the same data.
On Windows the launcher is `kg.cmd` (`"%LAYA_GRAPH_KG_ROOT%\kg.cmd"`, or
`"$LAYA_GRAPH_KG_ROOT/kg.cmd"` from Git Bash); the bootstrap picks it
automatically and needs a `python3` (python.org installs provide `py -3`).

Given the actual paper ID and an English excerpt of 60–1000 characters,
submit one JSON file with `paper_id`, `source_excerpt` and `policy: "auto"`:

```sh
"$LAYA_GRAPH_KG_ROOT/kg" select --request /path/to/request.json
```

`auto` uses exact source links first; otherwise code generates eight candidates,
calls the trained Laya worker and collects source evidence and conditions.
For several excerpts, send `{"items":[...requests...]}` with at most sixteen
items. Batch items are sequential; this is not tensor batching.

Read the selected evidence and applicability conditions against the user's
goal. Keep exact link sets as sets. Treat confidence and `needs_review` as
routing information, not truth. If evidence is inadequate, search the missing
concept or condition directly rather than automatically redoing the entire
graph. For general discovery, use `kg search ... --compact`, then source pages.

The validated task is English source-to-relationship alignment. Do not pass a
general user question as an invented source quotation or assume learned
STOP/BACK. Idle cleanup reuses one model and then releases owned resources
after fifteen unused minutes, deferring active leases/jobs/transactions.
