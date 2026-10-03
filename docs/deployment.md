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
The default backend is Laya. [Decision 2.0 setup](decision2.md) uses
`decision2.example.json`, `requirements-decision2.txt` and a separately provisioned
verified model package; model assets remain outside this repository.
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

## Windows and NVIDIA CUDA

The Python runtime and the Laya worker run natively on Windows 10/11 with an
NVIDIA GPU. Neo4j is still a Linux container, so Docker Desktop with its WSL 2
backend is required for the graph; the CLI starts Docker Desktop if it is
installed in the default location.

`install.ps1` performs the steps below and reports missing private assets.
Manual equivalent:

```powershell
py -3.12 -m venv knowledgegraph\.venv
knowledgegraph\.venv\Scripts\python -m pip install -r requirements-runtime-cuda.txt
py -3.12 -m venv .venv-model
.venv-model\Scripts\python -m pip install -r requirements-model-cuda.txt
python scripts\fetch_laya.py
knowledgegraph\kg.cmd prepare
```

- `kg.cmd` replaces `./kg` and enables Python UTF-8 mode; managed services
  receive `PYTHONUTF8=1` too. Run any project script directly with
  `PYTHONUTF8=1` (or `python -X utf8`), because JSON records are UTF-8.
- The `*-cuda.txt` files pin `torch==2.7.1+cu128`, whose kernels include
  Blackwell GPUs (RTX 50xx). Older GPUs work with the same wheels.
- `decision.json` `worker_python` may keep the POSIX form
  `../.venv-model/bin/python`; it maps to `Scripts\python.exe` on Windows.
- `decision.json` `device`: `auto` (CUDA, else MPS, else CPU), `cuda`, `cuda:1`,
  `mps` or `cpu`. An explicit device is enforced: Laya's silent CPU fallback
  (missing CUDA, out of memory) makes the worker fail to start instead.
- `decision.json` `precision` defaults to `fp32`. The head temperature was
  calibrated from fp32 one-row forwards, and Laya would otherwise autocast to
  bf16 on CUDA. `amp` is faster but must be re-validated against labels before
  the 0.95 review threshold is trusted.
- `settings.json` `embedding.device`: `cpu` reproduces the original float32
  BGE-M3 vectors exactly; `auto`/`cuda` embeds on the GPU (float32, numerically
  equivalent up to rounding).
- Graceful stop sends CTRL_BREAK to the service's own process group (uvicorn
  drains requests and runs lifespan cleanup); idle force-release terminates the
  owned process tree. Locks use `LockFileEx` with the same shared/exclusive
  lease semantics as `flock`.
- CUDA torch wheels contain header paths of about 130 characters. Keep the
  project root short (the venv path must stay under Windows' 260-character
  limit) or enable `LongPathsEnabled`; otherwise pip fails with
  `No such file or directory` for a `torch\include\ATen\ops\...` file.
- The first CUDA forward initializes kernels and takes about 15 s (RTX 5080);
  warm calls take about 15 ms. Warmup therefore uses
  `worker_load_timeout_seconds` instead of the per-request timeout.
- Paper ingestion calls Poppler's `pdftotext`; install a Windows Poppler build
  (for example `scoop install poppler` or `conda install poppler`) on `PATH`.
- macOS' `caffeinate` sleep guard for long ingestion has no Windows
  counterpart here; disable sleep in power settings for long builds.

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
