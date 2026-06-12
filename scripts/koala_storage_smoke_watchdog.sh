#!/usr/bin/env bash
set -u

cd "$(dirname "$0")/.."

uv sync --python "$(which python3)" --extra torch
source "${UV_PROJECT_ENVIRONMENT:-.venv}/bin/activate"
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

: "${ARCSTORE_SMOKE_MODE:=nomount}"
: "${ARCSTORE_SMOKE_TIMEOUT:=240s}"

timeout "${ARCSTORE_SMOKE_TIMEOUT}" python scripts/koala_storage_smoke.py
rc=$?

if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
    result="${ARCSTORE_SMOKE_S3%/}/result-${ARCSTORE_SMOKE_MODE}.json"
    if s5cmd ls "$result" >/dev/null 2>&1; then
        echo "ARCSTORE_SMOKE_WATCHDOG_EXIT result=${result} rc=${rc}"
        exit 0
    fi
fi

exit "$rc"
