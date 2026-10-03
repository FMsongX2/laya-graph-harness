# Laya Graph Harness

Integration kit for a local **Cognee + Neo4j + trained Laya** workflow.
The agent submits a bounded request once; code retrieves candidates, makes the
local choice, and collects source evidence and conditions. The agent evaluates
goal relevance and decides whether a focused follow-up is needed.

This first version supports **short English source excerpts → one paper's
reviewed relationship candidates**. It is not a general multi-hop navigator,
and it has no learned STOP/BACK policy. The trained checkpoint and the approved
knowledge corpus are separate local assets and are not in this repository.

## Included

- Read-only Neo4j candidate and evidence access with approved-record checks.
- Exact citation-link lookup before an optional eight-candidate Laya choice.
- Resident JSONL model worker, sequential batches of up to sixteen requests.
- Shared activity leases, singleton startup, one warmup per worker.
- Automatic cleanup after 15 minutes idle, busy-job/transaction protection,
  process ownership checks, and lazy restart on the next managed call.
- Agent skill/bootstrap source, installation overlay, contract tests, and
  sanitized local benchmark aggregates.

To browse reviewed knowledge in Obsidian (optionally beside PersonaGraph notes),
see [Obsidian view](docs/obsidian.md): `kg export-obsidian --out <vault>/knowledge`.

## Run the tests without models or private data

Python 3.12 on macOS or Linux:

```sh
python3.12 -m venv .venv-test
.venv-test/bin/python -m pip install -r requirements-test.txt
.venv-test/bin/python scripts/configure.py
.venv-test/bin/python -m unittest discover -s knowledgegraph/tests -p 'test_*.py' -v
.venv-test/bin/python -m unittest discover -s tests -p 'test_*.py' -v
.venv-test/bin/python scripts/check_export.py
```

The generated `knowledgegraph/config/settings.json` and `decision.json` are
ignored local files. Example configuration is sufficient for the synthetic
contract tests. It is not sufficient to start real inference or graph search.

## Use with an existing KnowledgeGraph deployment

The supported deployment path is an **offline code overlay onto an existing
configured project**, preserving its dataset, credentials, models and single
runtime owner. Do not start a second project root against the same registered
Neo4j containers while the original runtime/idle watcher is active.

```sh
python scripts/install_overlay.py --target "$KNOWLEDGEGRAPH_ROOT"
# Preview only. Stop the target's model/search services and idle watcher first.
python scripts/install_overlay.py --target "$KNOWLEDGEGRAPH_ROOT" --apply
"$KNOWLEDGEGRAPH_ROOT/kg" prepare
"$KNOWLEDGEGRAPH_ROOT/kg" select --request /path/to/request.json
"$KNOWLEDGEGRAPH_ROOT/kg" lifecycle-status
```

The installer refuses active leases, live managed processes and occupied
ports. It backs up replaced code and leaves settings, private keys, corpus and
weights intact. Publication of this repository did not install the overlay
or change the already-running local deployment.

See [deployment prerequisites](docs/deployment.md) for the required graph
schema, private asset layout and dependency versions. The bundled
`knowledgegraph/` source can also form a new deployment once these private
assets and configuration have been provisioned explicitly.

## Request and decision policy

```json
{
  "paper_id": "P99",
  "source_excerpt": "Actual English source text, between sixty and one thousand characters, obtained from the relevant paper.",
  "policy": "auto"
}
```

This is a schema illustration, not a real indexed quotation. `auto` follows
exact source links first, otherwise ranks eight paper-scoped candidates and
calls Laya once. `laya` deliberately bypasses the citation shortcut for tests.
Multiple exact links are returned as a set, without forcing one answer.

The output preserves quotations, applicability conditions and reviewed page/
table coordinates. `candidate`, `needs_review` and confidence are routing
signals, not scientific truth certificates. Read [agent fallback policy](docs/agent-policy.md).

## Measurements and practical limits

Measured locally on Apple M5 Pro / 48 GiB, 2026-10-03:

| Bounded task | Median | Label agreement |
|---|---:|---:|
| Codex directly selects and reads evidence, 10 cases | 12.334 s | 9/10 |
| Codex delegates to resident Laya, same 10 cases | 9.074 s | 10/10 |
| Code calls resident Laya, same 10 cases | 0.0768 s | 10/10 |
| Live Laya service, 150 previously evaluated excerpts | 0.104 s | 124/150 |

Agent runs used `gpt-6-luna / high`; they include fresh CLI startup, remote
turns and final output. Local service timing covers selection/evidence only.
These are retrospective bounded tests, not proof of general graph-navigation
quality or a replacement for the main agent. In the live set, 135/150 results
were marked for review. Exact citation lookup covered labeled relations in
150/150 sets with no model calls; set coverage is not single-choice accuracy.

Six concurrent preparations reused one search process, one model worker and
one idle watcher. Actual batch work and open Neo4j transactions blocked
cleanup. An aged activity marker exercised real timed shutdown and cold
recovery without altering the 900-second production policy.
See [sanitized evidence](verification/local-results.json).

## Repository boundaries

No source papers, approval records, database files, source excerpts from the
private evaluation corpus, raw agent traces, checkpoints, API credentials or
host hook trust state are tracked. The export check applies to every tracked
file and to the repository's publication review.

[Upstream notices](THIRD_PARTY_NOTICES.md) identify dependencies and pinned
Laya source. No project reuse license has been selected yet; source visibility
and an open-source license are separate choices.
