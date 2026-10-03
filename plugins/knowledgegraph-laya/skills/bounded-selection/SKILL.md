---
name: bounded-selection
description: Use a configured Cognee/Neo4j runtime with Laya or Decision 2.0 for bounded English source alignment and experimental evidence-feedback walks. General problem discovery uses existing graph search.
allowed-tools: Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/bootstrap.py)
---

# Bounded local graph selection

Preparation: !`python3 "${CLAUDE_SKILL_DIR}/scripts/bootstrap.py"`

Set `LAYA_GRAPH_KG_ROOT` to the single installed KnowledgeGraph project root.
Claude Code preprocesses the command before reading this skill. In other
hosts, use the bundled hook source or run `"$LAYA_GRAPH_KG_ROOT/kg" prepare`
once before querying. Do not start parallel project roots over the same data.

Given the actual paper ID and an English excerpt of 60–1000 characters,
submit one JSON file with `paper_id`, `source_excerpt` and `policy: "auto"`:

```sh
"$LAYA_GRAPH_KG_ROOT/kg" select --request /path/to/request.json
```

`auto` uses exact source links first; otherwise code generates eight candidates,
calls the configured local worker (trained Laya by default, optional Decision 2.0)
and collects source evidence and conditions. `policy: "model"` forces the current
backend; `policy: "laya"` explicitly requires Laya.
For several excerpts, send `{"items":[...requests...]}` with at most sixteen
items. Batch items are sequential; this is not tensor batching.

Read the selected evidence and applicability conditions against the user's
goal. Keep exact link sets as sets. Treat confidence and `needs_review` as
routing information, not truth. If evidence is inadequate, search the missing
concept or condition directly rather than automatically redoing the entire
graph. For general discovery, use `kg search ... --compact`, then source pages.

If the configured deployment enables the experimental walk, send a short English
goal with one actual `start_node_id` or `paper_id` to `kg walk --request ...`.
Code repeats local choices with evidence feedback and excludes visited nodes from
each next menu. Inspect the path, conditions and stop reason; bounded traversal
and target ID arrival are not semantic goal-completion certificates.

The validated task is English source-to-relationship alignment. Do not pass a
general user question as an invented source quotation or assume learned
STOP/BACK. Idle cleanup reuses one model and then releases owned resources
after fifteen unused minutes, deferring active leases/jobs/transactions.
