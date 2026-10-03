"""가벼운 로컬 CLI. 모델·Cognee는 상주 서비스에서 실행한다."""
from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import shutil
import sqlite3
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime.environment import environment
from runtime.lifecycle import tracked_cli, status as lifecycle_status

CONFIG = json.loads((ROOT / "config/settings.json").read_text())
BASE = f"http://127.0.0.1:{CONFIG['server']['port']}"
STATE = ROOT / "runtime/service-process.json"


def emit(value):
    print(json.dumps(value, ensure_ascii=False, indent=2))


def compact_search(result):
    """Project source-bearing matches without rephrasing claims or evidence."""
    output = {key: result[key] for key in ["query", "requested_mode", "answer_generated", "datasets"] if key in result}
    matches = []; seen = set()
    def visit(value):
        if isinstance(value, dict):
            if value.get("semantic_source_status"):
                matches.append({"kind": "unverified_indexed_text", **{key: value[key] for key in
                                ["id", "document_name", "text", "semantic_source_status"] if key in value}})
                return True
            assertion = value.get("semantic_assertion")
            if isinstance(assertion, dict):
                identity = ("assertion", assertion["id"], assertion.get("pdf_sha256"))
                if identity not in seen:
                    seen.add(identity)
                    matches.append({"kind": "semantic_assertion", **{key: item for key, item in assertion.items()
                                    if key != "retrieval_text"}})
                return True
            if isinstance(value.get("paper_source"), dict):
                identity = ("source", value.get("id"), value.get("text"))
                if identity not in seen:
                    seen.add(identity)
                    matches.append({"kind": "original_text", **{key: value[key] for key in
                                    ["id", "document_name", "chunk_index", "text", "paper_source"] if key in value}})
                return True
            return any([visit(item) for item in value.values()])
        elif isinstance(value, list):
            return any([visit(item) for item in value])
        return False
    unstructured = []
    for item in result.get("results", []):
        structured = visit(item)
        if not structured and item.get("context_result"):
            unstructured.append({"dataset": item.get("dataset_name"), "context": item["context_result"],
                                 "source_coordinates_attached": False})
    output["matches"] = matches
    # Graph-only retrieval may return a context without structured chunks.
    # Preserve it explicitly instead of pretending that the source is known.
    output["unstructured_contexts"] = unstructured
    return output


def request(path, data=None, timeout=3600):
    """로컬 API만 호출하고 실패 응답도 보존한다."""
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(BASE + path, data=body, headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError("Local service is unavailable. Run ./kg up first.") from exc


def owned_process():
    """보관 PID가 여전히 이 프로젝트 서비스인지 확인한다."""
    if not STATE.exists():
        return None
    state = json.loads(STATE.read_text())
    command = subprocess.run(["ps", "-p", str(state["pid"]), "-o", "command="],
                             capture_output=True, text=True).stdout.strip()
    return state if str(ROOT / "runtime") in command and "uvicorn" in command else None


def start():
    # Several agent sessions can reach the same lazy-start path concurrently.
    with (ROOT / 'runtime/service-start.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return start_locked()


def start_locked():
    """별도 프로세스 그룹으로 로컬 서비스를 시작하고 준비 상태를 확인한다."""
    if CONFIG.get("graph", {}).get("dataset_handler") == "neo4j_community":
        docker = shutil.which("docker")
        if not docker:
            raise RuntimeError("Neo4j requires the Docker CLI")
        def docker_ready():
            return subprocess.run([docker, "info", "--format", "{{.ServerVersion}}"],
                                  capture_output=True, timeout=10).returncode == 0
        if not docker_ready():
            if sys.platform == "darwin" and Path("/Applications/Docker.app").exists():
                subprocess.run(["open", "-a", "Docker"], check=True)
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    if docker_ready():break
                    time.sleep(1)
                else:raise RuntimeError("Docker Desktop did not become ready; start it and retry ./kg up")
            else:raise RuntimeError("Start the Docker daemon before ./kg up")
    state = owned_process()
    if state:
        return wait_ready(state)
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", CONFIG["server"]["port"])) == 0:
            raise RuntimeError("Configured port is already occupied by another process")
    command = [str(ROOT / ".venv/bin/python"), "-m", "uvicorn", "service:app", "--app-dir",
               str(ROOT / "runtime"), "--host", "127.0.0.1", "--port", str(CONFIG["server"]["port"]), "--workers", "1"]
    log_path = ROOT / "logs/service.log"
    log_path.parent.mkdir(exist_ok=True)
    with log_path.open("a") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=environment(), stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=log, start_new_session=True)
    state = {"pid": child.pid, "command": command, "log": str(log_path), "started_at": time.time()}
    STATE.write_text(json.dumps(state, indent=2) + "\n")
    return wait_ready(state)


def wait_ready(state):
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if not owned_process():
            raise RuntimeError(f"Service exited; see {state['log']}")
        try:
            health = request("/health", timeout=2)
            if health.get("ready"):
                return {"running": True, "pid": state["pid"], "health": health}
        except RuntimeError:
            pass
        time.sleep(.5)
    raise RuntimeError(f"Service is still starting; see {state['log']}")


def stop():
    """확인한 소유 프로세스만 정상 종료하며 공유 Ollama는 건드리지 않는다."""
    state = owned_process()
    if not state:
        return {"running": False}
    os.kill(state["pid"], signal.SIGTERM)
    for _ in range(60):
        if not owned_process():
            return {"running": False, "stopped_pid": state["pid"]}
        time.sleep(.25)
    raise RuntimeError("Shutdown is waiting for active requests; no force kill performed")


def prepare():
    """Ready shared services once; subsequent skill uses reuse the same processes."""
    import decision_cli
    with (ROOT / 'runtime/prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        started = time.perf_counter()
        knowledge = start()
        if knowledge['health'].get('knowledge_operation') in {'build', 'card-index', 'evidence-index', 'semantic-index'}:
            raise RuntimeError('KnowledgeGraph is building; prepare can resume after the existing build finishes')
        graph = request('/kg/graph-ready', timeout=190)
        decision_cli.start()
        model = decision_cli.call('/warmup', {}, timeout=40)
        if not graph.get('ready') or not model.get('ready') or not model.get('warmed'):
            raise RuntimeError('KnowledgeGraph/Laya preparation did not become ready')
        return {'ready': True, 'seconds': time.perf_counter() - started,
                'knowledge_pid': knowledge['pid'], 'embedding_loaded': knowledge['health']['embedding_loaded'],
                'graph': graph, 'decision': model,
                'resident_until_idle_timeout': True, 'lifecycle': lifecycle_status(), 'full_graph_materialized': False,
                'corpus_indexing': False, 'external_model_calls': 0}


def paper_status():
    script = "build_cards.py" if CONFIG["knowledge"]["strategy"] == "knowledge_cards" else "index_fulltext.py"
    job = ROOT / ("state/fulltext-ingestion" if script == "index_fulltext.py" else "state/card-ingestion")
    progress = job / "progress.json"
    process_file = job / "worker-process.json"
    state = json.loads(process_file.read_text()) if process_file.exists() else None
    alive = False
    if state:
        command = subprocess.run(["ps", "-p", str(state["pid"]), "-o", "command="],
                                 capture_output=True, text=True).stdout
        alive = str(ROOT / "tools" / script) in command
    return {"worker_running": alive, "process": state,
            "progress": json.loads(progress.read_text()) if progress.exists() else None}


def start_papers(files=None):
    state = paper_status()
    if state["worker_running"]:
        return state
    start()
    script = "build_cards.py" if CONFIG["knowledge"]["strategy"] == "knowledge_cards" else "index_fulltext.py"
    job = ROOT / ("state/fulltext-ingestion" if script == "index_fulltext.py" else "state/card-ingestion")
    job.mkdir(parents=True, exist_ok=True)
    command = [str(ROOT / ".venv/bin/python"), str(ROOT / "tools" / script), *(files or [])]
    with (job / "worker.log").open("a") as log:
        worker = subprocess.Popen(command, cwd=ROOT, env=environment(), stdin=subprocess.DEVNULL,
                                  stdout=log, stderr=log, start_new_session=True)
    process = {"pid": worker.pid, "command": command, "log": str(job / "worker.log")}
    if sys.platform == "darwin" and Path("/usr/bin/caffeinate").exists():
        guard = subprocess.Popen(["/usr/bin/caffeinate", "-i", "-w", str(worker.pid)],
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, start_new_session=True)
        process["idle_sleep_guard_pid"] = guard.pid
    (job / "worker-process.json").write_text(json.dumps(process, indent=2) + "\n")
    return {"worker_started": True, "process": process,
            "status_command": str(ROOT / "kg") + " papers-status"}


@tracked_cli
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["up", "down", "status", "doctor", "datasets", "ingest-papers", "papers-status", "semantic-status", "neo4j-info", "decision-up", "decision-down", "decision-status", "prepare", "lifecycle-status"]:
        commands.add_parser(name)
    select_cmd = commands.add_parser("select", help="코드가 Neo4j 후보 조회·Laya 선택·근거 수집을 한 번에 수행")
    select_cmd.add_argument("--request", help="단일 또는 items 배치 JSON 파일; -는 stdin")
    select_cmd.add_argument("--paper", help="논문 범위, 예: P03")
    select_cmd.add_argument("--excerpt-file", help="영어 원문 60~1000자 파일; -는 stdin")
    select_cmd.add_argument("--policy", choices=["auto", "laya"], default="auto")
    select_cmd.add_argument("--include-candidates", action="store_true")
    export = commands.add_parser("export-obsidian", help="승인된 근거를 읽기 전용 Obsidian 노트로 내보내기")
    export.add_argument("--out", required=True, help="빈 폴더 또는 이전 내보내기 폴더 (예: <vault>/knowledge)")
    cards = commands.add_parser("build-cards", help="원문 근거 검색층과 정교한 지식 카드 그래프 구축")
    cards.add_argument("files", nargs="*")
    source = commands.add_parser("source", help="모델 호출 없이 원문 PDF 페이지 직접 읽기")
    source.add_argument("paper_id")
    source.add_argument("--page", type=int, required=True)
    add = commands.add_parser("add", help="원문 등록; 그래프 구축은 별도 build")
    add.add_argument("files", nargs="*")
    add.add_argument("--text")
    add.add_argument("--dataset", default=CONFIG["knowledge"]["evidence_dataset"])
    add.add_argument("--domain", action="append", default=[])
    build = commands.add_parser("build", help="등록 자료의 그래프·임베딩을 증분 구축")
    build.add_argument("--dataset", default=CONFIG["default_dataset"])
    build.add_argument("--chunk-size", type=int, default=CONFIG["cognee"]["chunk_size"])
    build.add_argument("--dry-run", action="store_true")
    search = commands.add_parser("search", help="최종 답변 생성 없이 지식과 근거 조회")
    search.add_argument("query")
    search.add_argument("--dataset", action="append")
    search.add_argument("--domain", action="append", default=[])
    search.add_argument("--layer", choices=["source", "evidence"], default="source")
    search.add_argument("--top-k", type=int, default=10)
    search.add_argument("--mode", choices=["hybrid", "graph", "chunks"], default="hybrid")
    search.add_argument("--compact", action="store_true", help="중복 프롬프트를 빼고 관계·조건·원문 근거만 표시")
    args = parser.parse_args()
    try:
        if args.command == "up": result = start()
        elif args.command == "prepare": result = prepare()
        elif args.command == "lifecycle-status": result = lifecycle_status()
        elif args.command == "down": result = stop()
        elif args.command.startswith("decision-"):
            import decision_cli
            result = {"decision-up": decision_cli.start, "decision-down": decision_cli.stop,
                      "decision-status": decision_cli.status}[args.command]()
        elif args.command == "select":
            import decision_cli
            if args.request:
                if args.paper or args.excerpt_file or args.include_candidates or args.policy != "auto":
                    raise ValueError("Use --request alone, or --paper with --excerpt-file")
                payload = json.loads(sys.stdin.read() if args.request == "-" else Path(args.request).read_text())
            else:
                if not args.paper or not args.excerpt_file:
                    raise ValueError("Provide --request JSON, or --paper and --excerpt-file")
                text = sys.stdin.read() if args.excerpt_file == "-" else Path(args.excerpt_file).read_text()
                payload = {"paper_id": args.paper.upper(), "source_excerpt": text,
                           "policy": args.policy, "include_candidates": args.include_candidates}
            if not isinstance(payload, dict): raise ValueError("Selection request must be a JSON object")
            if not owned_process():
                start()
                request('/kg/graph-ready', timeout=190)
            result = decision_cli.selection(payload)
        elif args.command == "ingest-papers": result = start_papers()
        elif args.command == "build-cards":
            raise RuntimeError("Knowledge-card generation is inactive; the selected strategy is complete original text")
        elif args.command == "papers-status": result = paper_status()
        elif args.command == "export-obsidian":
            from export_obsidian import export
            result = export(args.out)
        elif args.command == "semantic-status":
            result = json.loads((ROOT / "state/semantic-ingestion/progress.json").read_text())
        elif args.command == "source":
            catalog = json.loads((ROOT / "data/hackerton-papers/catalog.json").read_text())
            record = catalog["records"][args.paper_id]
            path = ROOT / "data/hackerton-papers" / record["pages_path"]
            flow = path.with_name("flow-pages-v1.jsonl")
            if flow.exists() and args.paper_id != "P42": path = flow
            pages = [json.loads(line) for line in path.read_text().splitlines()]
            page = next(page for page in pages if page["page"] == args.page)
            result = {"paper_id": args.paper_id, "title": record["title"], "pdf_page": args.page,
                      "original_pdf": str(ROOT / "data/hackerton-papers" / record["pdf_path"]),
                      "text": page["text"]}
        elif args.command == "status":
            state = owned_process()
            result = {"running": bool(state), "process": state,
                      "health": request("/health", timeout=3) if state else None}
        elif args.command == "doctor":
            result = {"root": str(ROOT), "python": sys.version.split()[0],
                      "cognee": importlib.metadata.version("cognee"), "settings": CONFIG,
                      "embedding_snapshot_exists": Path(CONFIG["embedding"]["snapshot"]).is_dir(),
                      "service_running": bool(owned_process())}
        elif args.command == "neo4j-info":
            connection = sqlite3.connect(f"file:{ROOT}/state/system/databases/cognee_db?mode=ro", uri=True)
            result = {"datasets": [{"name": name, "uri": uri, "database": database,
                         "handler": handler, "container": json.loads(info).get("container_name")}
                       for name, uri, database, handler, info in connection.execute(
                         "SELECT d.name,b.graph_database_url,b.graph_database_name,b.graph_dataset_database_handler,b.graph_database_connection_info "
                         "FROM dataset_database b JOIN datasets d ON d.id=b.dataset_id WHERE b.graph_database_provider='neo4j'")],
                      "credentials": "Encrypted in the dataset registry; encryption key is kept in the private configured key file."}
            connection.close()
        elif args.command == "datasets":
            if not owned_process(): start()
            result = request("/kg/datasets")
        elif args.command == "add":
            if not owned_process(): start()
            files = [str(Path(path).expanduser().resolve()) for path in args.files]
            result = request("/kg/add", {"files": files, "text": args.text, "dataset": args.dataset, "domains": args.domain})
        elif args.command == "build":
            if not owned_process(): start()
            result = request("/kg/build", {"dataset": args.dataset, "chunk_size": args.chunk_size, "dry_run": args.dry_run})
        else:
            if not owned_process(): start()
            health = request("/health", timeout=5)
            if health.get("knowledge_operation") in {"build", "card-index", "evidence-index", "semantic-index"}:
                raise RuntimeError("KnowledgeGraph is building. Check ./kg semantic-status and ./kg papers-status; "
                                   "source papers remain readable under " + str(ROOT / "data/hackerton-papers/cognee-inputs"))
            modes = {"hybrid": "HYBRID_COMPLETION", "graph": "GRAPH_COMPLETION", "chunks": "CHUNKS"}
            layer = CONFIG["knowledge"]["evidence_dataset"] if args.layer == "evidence" else CONFIG["default_dataset"]
            result = request("/kg/search", {"query": args.query, "datasets": args.dataset or [layer],
                             "domains": args.domain, "top_k": args.top_k,
                             "mode": "CHUNKS" if args.layer == "evidence" or CONFIG["knowledge"]["strategy"] == "fulltext" else modes[args.mode]})
            if args.compact:
                result = compact_search(result)
        emit(result)
    except (RuntimeError, OSError, ValueError) as exc:
        emit({"error": str(exc)})
        raise SystemExit(1)


if __name__ == "__main__":
    main()
