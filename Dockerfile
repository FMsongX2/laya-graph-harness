# syntax=docker/dockerfile:1.7
# Laya graph harness runtime: Cognee/BGE-M3 service, resident Laya worker and the kg CLI.
# Private assets are mounted at /data/laya (see docs/docker.md); none are built in.
ARG PYTHON_IMAGE=python:3.12-slim-bookworm
FROM docker:29.8.2-cli AS docker-cli

FROM ${PYTHON_IMAGE}
# cu128 = NVIDIA (includes Blackwell sm_120 kernels); cpu = no GPU.
ARG TORCH_FLAVOR=cu128
ENV PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    APP=/opt/laya-graph-harness LAYA_HOME=/data/laya
ENV PATH=${APP}/knowledgegraph:${PATH}

RUN apt-get update \
 && apt-get install -y --no-install-recommends git poppler-utils tini ca-certificates \
 && rm -rf /var/lib/apt/lists/*
# Cognee manages per-dataset Neo4j containers through the host daemon (mounted socket).
COPY --from=docker-cli /usr/local/bin/docker /usr/local/bin/docker

WORKDIR ${APP}
COPY requirements*.txt ./
# One torch build in the base interpreter, shared by both environments (the wheels are ~4 GB).
RUN --mount=type=cache,target=/root/.cache/pip \
    TORCH=$(sed -n 's/^torch==//p' requirements-model.txt) \
 && pip install --extra-index-url https://download.pytorch.org/whl/${TORCH_FLAVOR} "torch==${TORCH}+${TORCH_FLAVOR}"
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv --system-site-packages knowledgegraph/.venv \
 && knowledgegraph/.venv/bin/pip install -r requirements-runtime.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv --system-site-packages .venv-model \
 && .venv-model/bin/pip install -r requirements-model.txt

COPY scripts/fetch_laya.py scripts/
RUN python scripts/fetch_laya.py && rm -rf model-worker/vendor/laya/.git

COPY . .
# Windows checkouts with core.autocrlf could carry CRLF into the shell launchers.
RUN sed -i 's/\r$//' knowledgegraph/kg docker/entrypoint.sh \
 && chmod +x knowledgegraph/kg docker/entrypoint.sh \
 && python -m compileall -q knowledgegraph/runtime knowledgegraph/tools model-worker

VOLUME ["/data/laya"]
ENTRYPOINT ["tini", "--", "/opt/laya-graph-harness/docker/entrypoint.sh"]
CMD ["serve"]
