# Docker deployment

The image contains authored code, both Python environments and the pinned Laya
source. It never contains papers, approval records, dataset state, credentials
or weights: `.dockerignore` is an allow-list, and `tests/test_install.py`
checks that private paths stay out of the build context.

## Requirements

- Docker Engine with Compose v2.24+ (Linux), or Docker Desktop with the WSL 2
  backend (Windows) / Docker Desktop (macOS, CPU only).
- NVIDIA GPU: driver 570+ and the NVIDIA Container Toolkit (Linux). Docker
  Desktop on Windows passes the GPU through WSL 2 without extra setup.
- Disk: about 9 GB for the CUDA image, 3 GB for the CPU image.

## Private assets volume

Everything private lives in one host directory, `LAYA_HOME` (default
`./laya-home`, ignored by Git), mounted at `/data/laya`:

```text
laya-home/
  config/
    settings.json          # seeded from the example on first start; edit it
    decision.json          # seeded from the example on first start
    neo4j-secret.json      # encryption key for the dataset registry
  state/                   # Cognee state, incl. system/databases/cognee_db
  data/semantic-assertions/  # approved source records
  models/bge-m3/           # embedding snapshot (settings.embedding.snapshot)
  model-worker/
    base-model/            # Laya base snapshot
    trained/               # head.safetensors + head-config.json
  logs/
```

Copying an existing native deployment: its `knowledgegraph/{state,data,models}`
and `config/*.json` map one-to-one onto these folders.

## Run

```sh
docker compose build                       # NVIDIA image (torch cu128)
docker compose run --rm harness check      # what is still missing in laya-home
docker compose up -d                       # links assets, builds the Neo4j image, kg prepare
docker compose exec harness kg select --request /data/laya/requests/example.json
docker compose exec harness kg lifecycle-status
docker compose down                        # releases services and owned Neo4j containers
```

CPU-only hosts use the override: `docker compose -f compose.yaml -f compose.cpu.yaml up -d`.
`LAYA_PREPARE=0` skips the eager `kg prepare`; services then start on first use.

## How it fits together

- **Neo4j:** Cognee starts one Neo4j container per dataset through the host
  daemon (mounted Docker socket), publishes it on the host's `127.0.0.1` and
  stores `bolt://localhost:<port>` in the registry. The harness therefore uses
  `network_mode: host`, so the registered URLs work unchanged. The configured
  Neo4j/APOC image is built on first start if the daemon lacks it.
- **Ollama** (`kg search`/`kg build` only; `kg select` does not use it): run it
  on the host, or start the bundled service with
  `docker compose --profile ollama up -d`. With Docker Desktop, a host-installed
  Ollama is reached at `http://host.docker.internal:11434`; set
  `chat.upstream` in `settings.json` accordingly.
- **Lifecycle:** unchanged from native runs. Services start lazily, idle out
  after `lifecycle.json`'s timeout and restart on the next `kg` call. Stopping
  the container runs the same lease-respecting cleanup, so it never interrupts
  active work and stops only the Neo4j containers this registry owns.
- **GPU:** `decision.json` `device`/`precision` and `settings.json`
  `embedding.device` apply as in [deployment](deployment.md#windows-and-nvidia-cuda).
