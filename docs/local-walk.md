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
