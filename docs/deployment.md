# Deployment prerequisites

The current adapter expects the existing approved Cognee data contract:

- `config/settings.json` selects a Neo4j-backed semantic dataset.
- `state/system/databases/cognee_db` registers dataset connection metadata.
- A private `config/neo4j-secret.json` (or the configured local path) decrypts
  the registry password. Never place it in Git.
- Neo4j has the reified source assertion graph: source entity → assertion →
  target, assertion → reviewed paper, conditions, and evidence → source paper.
- `data/semantic-assertions/Pxx/Pxx-Axxx.json` contains approved source records
  matching the graph's relation, PDF hash, quotation and conditions.
- Local BGE-M3 dense assets and an existing Ollama model support the original
  Cognee search service. Set their real locations/name in ignored local config.
- The separate model environment has the pinned Laya source, base assets and
  task-trained checkpoint. Base Laya alone did not validate this task profile.

`knowledgegraph/config/*.example.json` describes configuration structure.
`scripts/configure.py` creates ignored local copies without starting anything.
The standalone snapshot's default model layout is:

```text
model-worker/
  worker.py
  build_data.py
  vendor/laya/              # ignored pinned upstream checkout
  base-model/               # ignored local base snapshot
  trained/
    head.safetensors        # ignored trained checkpoint
    head-config.json        # ignored temperature/config, colocated with weights
.venv-model/                # ignored Python 3.12 environment
```

`python scripts/fetch_laya.py` obtains only pinned upstream code. Install its
dependencies into `.venv-model` using `requirements-model.txt`; model assets
must be supplied separately. Tested base revision is listed in the upstream
notices. The task profile uses max length 1536, head length 512, temperature
2.5 and eight candidates. It was full-encoder/head supervised adaptation,
not official RLCD training. The original trained artifact's SHA-256 was
`292ee37598b46d9a883839a6bf8766b2142bd1443180366f7fe27fda3be97044`.

The server environment uses `requirements-runtime.txt`. The existing project
overlay retains its configured environment and assets. Build the pinned
Neo4j/APOC image with its included Dockerfile if provisioning that dependency:

```sh
docker build -t knowledgegraph-neo4j:5.26.31-local knowledgegraph/config/neo4j
```

Creating, restoring or migrating a dataset is a separate operation. This
repository does not include private state, a paper corpus or trained weights,
and an empty clone cannot reproduce the reported empirical accuracy.

For skill use, set `LAYA_GRAPH_KG_ROOT` to the single installed project root.
The skill source is `plugins/knowledgegraph-laya/skills/bounded-selection`.
Claude Code's dynamic preprocessing calls the bundled bootstrap. Codex can
use the bootstrap as `UserPromptSubmit`/`PreToolUse` hook source, but has no
dedicated SkillLoaded event. Registration and exact-definition trust remain
explicit host setup steps; no hook trust hashes are distributed or installed.

Idle configuration lives in the installed project's `config/lifecycle.json`.
Polling health does not keep models alive. Managed CLI/API work holds leases;
long direct scripts can use `runtime.lifecycle.Lease`. Unmanaged clients are
not assumed to expose all future activity to the idle controller.
