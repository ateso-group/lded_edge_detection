#!/usr/bin/env bash
# =============================================================================
# LDED — Dataset Download (Wrapper)
# Delegiert an download_datasets.py, welches automatisch ein venv erstellt.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${1:-./data/raw}"

exec python3 "$SCRIPT_DIR/download_datasets.py" --data-root "$DATA_ROOT"
