#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ARPO_PYTHON="${ARPO_PYTHON:-python3}"
extras="mlx,reports"
if [[ "${1:-}" == "--no-mlx" ]]; then extras="reports"; fi
if command -v uv >/dev/null 2>&1; then
  uv venv --python "$ARPO_PYTHON" .venv
  uv pip install --python .venv/bin/python -e ".[${extras}]"
else
  "$ARPO_PYTHON" -c 'import sys; assert sys.version_info >= (3, 11), "Python 3.11+ required"'
  "$ARPO_PYTHON" -m venv .venv
  .venv/bin/python -m pip install -e ".[${extras}]"
fi
printf 'Готово. Выполните source .venv/bin/activate, затем arpo --help\n'
