#!/bin/sh
# Native install on macOS/Linux: picks Python 3.12 and runs scripts/install.py.
set -eu
cd "$(dirname "$0")"
for candidate in "${PYTHON:-}" python3.12 python3; do
  [ -n "$candidate" ] && command -v "$candidate" >/dev/null 2>&1 && exec "$candidate" scripts/install.py --python "$(command -v "$candidate")" "$@"
done
echo "Python 3.12 not found; set PYTHON=/path/to/python3.12" >&2
exit 1
