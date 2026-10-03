#!/bin/sh
# Map the private deployment volume onto the project layout, then run a command.
#   serve (default)  link assets, build the Neo4j image if missing, kg prepare when ready, stay up
#   check            report missing private assets and exit
#   anything else    run it (e.g. `kg select --request /data/laya/requests/a.json`)
set -eu
KG="$APP/knowledgegraph"
PY="$KG/.venv/bin/python"

# link <project path> <volume path>: the project path becomes a symlink into the volume.
link() {
  mkdir -p "$(dirname "$2")"
  if [ -e "$1" ] && [ ! -L "$1" ]; then rm -rf "$1"; fi
  ln -sfn "$2" "$1"
}

mkdir -p "$LAYA_HOME/config" "$LAYA_HOME/model-worker/base-model" "$LAYA_HOME/model-worker/trained"
for name in settings decision; do
  # First start: seed editable configs from the examples; never overwrite existing ones.
  [ -f "$LAYA_HOME/config/$name.json" ] || cp "$KG/config/$name.example.json" "$LAYA_HOME/config/$name.json"
done
# Every file in the volume's config/ (settings, decision, Neo4j key, optional overrides).
for file in "$LAYA_HOME"/config/*; do
  [ -f "$file" ] && link "$KG/config/$(basename "$file")" "$file"
done
for dir in state data models logs; do
  mkdir -p "$LAYA_HOME/$dir"; link "$KG/$dir" "$LAYA_HOME/$dir"
done
link "$APP/model-worker/base-model" "$LAYA_HOME/model-worker/base-model"
link "$APP/model-worker/trained" "$LAYA_HOME/model-worker/trained"

case "${1:-serve}" in
  serve)
    "$PY" "$APP/scripts/install.py" neo4j-image || true
    if "$PY" "$APP/scripts/install.py" check && [ "${LAYA_PREPARE:-1}" = 1 ]; then
      kg prepare || echo "kg prepare failed; inspect $LAYA_HOME/logs" >&2
    fi
    # Services start lazily on the next kg call and stop after the idle timeout.
    # On container stop, release them (and the owned Neo4j containers) through the same
    # lease-respecting cleanup; it declines while work is active.
    trap 'cd "$KG" && "$PY" -c "from runtime import lifecycle; print(lifecycle.cleanup_if_idle(idle_timeout=0))"; exit 0' TERM INT
    echo "Ready for: docker compose exec harness kg select --request /data/laya/<request>.json"
    sleep infinity &
    wait
    ;;
  check)
    exec "$PY" "$APP/scripts/install.py" check
    ;;
  *)
    exec "$@"
    ;;
esac
