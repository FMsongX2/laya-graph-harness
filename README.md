# Laya Graph Harness

Integration kit for a local **Cognee + Neo4j + trained Laya** workflow, with an
optional local **Decision 2.0** worker.
The agent submits a bounded request once; code retrieves candidates, makes the
local choice, and collects source evidence and conditions. The agent evaluates
goal relevance and decides whether a focused follow-up is needed.

The default selector supports **short English source excerpts → one paper's
reviewed relationship candidates**. A separate experimental `kg walk` endpoint
chains local selections over actual semantic edges with request-local visited
filtering. It is bounded single-path exploration, not a validated general
navigator, and has no learned STOP/BACK policy. The trained checkpoint and the
approved knowledge corpus are separate local assets and are not in this repository.

## Included

- Read-only Neo4j candidate and evidence access with approved-record checks.
- Exact citation-link lookup before an optional eight-candidate Laya choice.
- Resident JSONL model worker, sequential batches of up to sixteen requests.
- Configurable local backend: Laya by default, or a separately provisioned
  Decision 2.0 package with backend identity and provider-specific review scope.
- Experimental local walk with visited UUID hash sets, candidate exclusion before
  model input, evidence feedback, bounded termination and isolated request state.
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

The [optional Decision 2.0 backend](docs/decision2.md) uses its own model environment
and verified local package. Select it in the installed project's ignored config;
the offline overlay does not change an existing deployment's backend or weights.

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
calls the configured local worker once (Laya by default). `model` deliberately
bypasses the citation shortcut; legacy `laya` explicitly requests that backend.
Multiple exact links are returned as a set, without forcing one answer.

The output preserves quotations, applicability conditions and reviewed page/
table coordinates. `candidate`, `needs_review` and confidence are routing
signals, not scientific truth certificates. Read [agent fallback policy](docs/agent-policy.md).

For iterative exploration, submit a short English goal and an actual start node
or paper ID through `kg walk`. Read the [loop and visited filtering contract](docs/local-walk.md).
`knowledgegraph/examples/walk-request.json` is a synthetic schema example, not
an indexed paper. The same resident worker consumes each new state after the
selected relation's evidence is read; there is no intermediate remote agent turn.

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

Additional server, Mac and whole-agent comparisons were completed on 2026-10-04.
Sol 2B on native MPS FP32 matched all 278 server choices and reached the six bounded
walk goals. Its live graph walk median was 1.133 s, source inference 0.489 s, and
general-choice inference 1.021 s. These are warm local timings.

On the same six goals, a separate whole-agent Sol trial was 16.799 → 13.257 s
(about 21.1% less time, 6/6 both). Four cases were faster and two slower. The Laya
trial was 15.127 → 9.897 s (34.6% less time), with delegated goal arrival 3/6.
These separate trials do not establish a universal quality-preserving speedup.
See [measurement scopes and full tables](docs/measurements.md) and
[sanitized Decision 2.0 aggregates](verification/decision2-results.json).

## Repository boundaries

No source papers, approval records, database files, source excerpts from the
private evaluation corpus, raw agent traces, checkpoints, API credentials or
host hook trust state are tracked. The export check applies to every tracked
file and to the repository's publication review.

[Upstream notices](THIRD_PARTY_NOTICES.md) identify dependencies and pinned
Laya source. No project reuse license has been selected yet; source visibility
and an open-source license are separate choices.
