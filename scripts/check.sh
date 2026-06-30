#!/usr/bin/env bash
# scripts/check.sh - Run all quality gates locally via uv, mirroring CI.
#
# Usage:
#   ./scripts/check.sh           # check-only
#   ./scripts/check.sh --fix     # apply safe Ruff fixes, then check
#   ./scripts/check.sh --fast    # skip dependency audit
#
# Exit code: 0 if all gates pass, 1 if any gate fails.
# Each gate is run independently so all failures are visible in one pass.

set -euo pipefail

FIX=false
FAST=false
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FAILED_GATES=()
REQ_FILE=""

for arg in "$@"; do
    case "$arg" in
        --fix) FIX=true ;;
        --fast) FAST=true ;;
        --help|-h)
            echo "Usage: ./scripts/check.sh [--fix] [--fast]"
            echo ""
            echo "  --fix   Apply safe Ruff fixes before checking"
            echo "  --fast  Skip pip-audit, useful offline or on slow machines"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg (use --help for usage)" >&2
            exit 1
            ;;
    esac
done

gate() {
    local label="$1"
    shift

    printf "\n--- %s ---\n" "$label"
    printf "  %s\n" "$*"
    if "$@"; then
        printf "  ok %s\n" "$label"
    else
        printf "  no %s\n" "$label"
        FAILED_GATES+=("$label")
    fi
}

cleanup() {
    if [[ -n "$REQ_FILE" && -f "$REQ_FILE" ]]; then
        rm -f "$REQ_FILE"
    fi
}
trap cleanup EXIT

dependency_audit() {
    REQ_FILE="$(mktemp)"
    uv export --format requirements-txt --no-hashes --output-file "$REQ_FILE" >/dev/null
    uv run pip-audit -r "$REQ_FILE" --desc off --progress-spinner off
}

mypy_targets() {
    local targets=("polars_ti")
    if compgen -G "scripts/*.py" >/dev/null; then
        targets+=("scripts")
    fi
    uv run mypy --strict "${targets[@]}"
}

pandas_purge_check() {
    ! grep -R -n -E "import pandas|from pandas|\bpd\." polars_ti --include="*.py"
    ! grep -R -n -E "import pandas|from pandas|\bpd\." tests --include="*_polars.py"
    ! grep -R -n -E "import pandas|from pandas|\bpd\." tests/conftest.py
}

cd "$PROJECT_ROOT"

printf "\nQuality gates - %s\n" "$(date '+%Y-%m-%d %H:%M:%S')"
printf "Project root: %s\n" "$PROJECT_ROOT"
[[ "$FIX" == "true" ]] && printf "Auto-fix mode enabled (--fix)\n"
[[ "$FAST" == "true" ]] && printf "Fast mode enabled (--fast)\n"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required. Install it from https://docs.astral.sh/uv/." >&2
    exit 1
fi

gate "uv lock check" \
    uv lock --check

if [[ "$FIX" == "true" ]]; then
    gate "Ruff lint fixes" \
        uv run ruff check polars_ti tests scripts --fix

    gate "Ruff format apply" \
        uv run ruff format polars_ti tests scripts
fi

gate "Ruff lint" \
    uv run ruff check polars_ti tests scripts

gate "Ruff format" \
    uv run ruff format --check polars_ti tests scripts

gate "Mypy strict" \
    mypy_targets

gate "Syntax check" \
    uv run python -m compileall -q polars_ti tests scripts

gate "Runtime pandas purge" \
    pandas_purge_check

gate "Pytest + coverage report" \
    uv run --extra test pytest tests/ -q --tb=short --cov=polars_ti --cov-report=term-missing --cov-fail-under=90

if [[ "$FAST" == "true" ]]; then
    printf "\n-- pip-audit (--fast skipped)\n"
else
    gate "Runtime dependency audit" dependency_audit
fi

if [[ "${#FAILED_GATES[@]}" -gt 0 ]]; then
    printf "\nFailed gates:\n"
    for gate_name in "${FAILED_GATES[@]}"; do
        printf "  - %s\n" "$gate_name"
    done
    exit 1
fi

printf "\nAll gates passed.\n"
