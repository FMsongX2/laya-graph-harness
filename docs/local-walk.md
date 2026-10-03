# Experimental local graph walk

`kg walk --request request.json` / `POST /walk` repeats a local loop:

```text
current node -> reviewed relation candidates -> Laya choice
  -> verify/read evidence and conditions -> build the next observed state -> repeat
```

Use an actual graph ID in `start_node_id`, or a real `paper_id`, and a short English
`goal`. `max_hops` defaults to 4 and is bounded at 16. `target_node_id` is optional.
`include_candidates` exposes the per-hop model payload for inspection. The normal
single-step selector API and trained checkpoint remain unchanged.

Visited relation and destination IDs live in a fresh Python hash set per request.
Neo4j excludes them before its result limit; Python filters again before building
the JSON choice menu. Duplicate relationship candidates are also removed.
No database property is written. The next input keeps the original goal, current
node and recently read evidence/conditions, rather than recycling a choice ID alone.
A single eligible candidate is followed without unnecessary inference.

An entity hop follows its reified assertion to the semantic destination. A paper
start first seeds from that paper's assertions; an assertion start resolves its
verified target first. These logical hops can comprise multiple physical edges.
Code stops at the requested target ID, lack of unvisited candidates or the hop cap.
ID arrival does not certify semantic goal completion; outputs always preserve
`goal_verified: false` and `needs_review: true`.

The existing model was adapted for eight source-relationship candidates. Changing
goals, history and menu cardinality is experimental and is not calibrated by that
profile. There is no learned STOP/BACK, no exhaustive frontier search, and no
general route-quality claim. Strict visited-target filtering can suppress a new
relation toward a previously visited concept. The main agent evaluates the
returned path, evidence and applicability conditions.

Eight isolated walk tests cover cycles, duplicate menus, evidence feedback,
request isolation, budgets, target termination, output errors and assertion seeds.
The local installed deployment also exercised six structurally chosen real starts:
one to two hops, pipeline median 0.152 s, HTTP median 0.162 s. One two-hop case used
two Laya calls; graph counts and evidence coordinates were checked. These figures
exclude cold loading and do not establish goal relevance or superiority over the
earlier single-step timing. No private graph IDs, source quotations or traces are
included in this reusable package.

An additional paired pilot on 2026-10-04 used six predeclared two-hop goals from
the installed graph. Both arms shared production candidate ranking, visited
filtering, current-node and recent-evidence payloads, and a two-hop limit. The
direct agent continued within one session across hops. The delegated agent made
one tool call for the entire local walk. Both used `gpt-6-luna/high` with warm
local resources; agent times include fresh CLI startup and identifier-only final
output. Gold routes were held only by the evaluator and were not injected into
runtime candidates or `target_node_id`.

| Arm | Median elapsed time | Goal-evidence arrival |
| --- | ---: | ---: |
| Agent chooses each step | 15.127 s | 6/6 |
| Agent delegates the full walk | 9.897 s | 3/6 |
| Warm local walk alone | 0.144 s pipeline / 0.148 s HTTP | 3/6 |

All three misses had the correct candidate available but selected a different
second-step relationship. The arm-median time reduction was 34.6%; this is not a
quality-preserving speedup over all tasks. In the three cases where both arms
reached the goal, paired time savings had a median of 2.763 s (range 0.432–9.656 s).
The 18 local runs repeated the same paths, and all 12 agent protocols were valid.
Live node/edge counts remained unchanged.

This supports the latency benefit of eliminating intermediate remote choices
within this pilot. It also shows that the current source-to-relationship model
needs improved goal-navigation quality before promoting the walk to a general
default. Keep the walk experimental. The pilot used manually authored goals
that name the intermediate method, one agent run per case/arm with alternating
order, and three local repeats. It does not establish general accuracy,
statistical significance, discovery quality, final-answer quality or cold-start
performance. Raw private corpus/graph traces remain in the installed project.

## Model replacement boundary

Keep the traversal loop when assessing a future decision model. `WalkEngine`
receives its graph and worker as constructor dependencies. The loop owns candidate
generation, visited state, evidence reads and termination; the worker owns model
loading and `evaluate(payload)`. A replacement adapter can translate the same
goal/current-node/recent-evidence/choice-menu input to the new provider and return
the selected offered key, probabilities and truncation information in the current
worker contract. Graph IDs and evidence provenance remain under the loop's control.

The installed provider is still Laya. Connecting a future model requires checking
its actual API and output contract, then rerunning matched goal/evidence and
latency comparisons. If an adapter makes remote calls, update the current
local-only call counts and timing metadata to reflect that execution. Preserve
this substitution point without committing the architecture to an unreleased
model or presuming that its latency or navigation quality will be better.
