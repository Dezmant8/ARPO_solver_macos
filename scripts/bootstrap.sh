#!/usr/bin/env bash
# One-time local setup. No modifications to shell profiles or system Python.
set -euo pipefail
cd "$(dirname "$0")/.."
ARPO_ROOT="$PWD"
ARPO_ECO_DIR="${ARPO_ECO_DIR:-$HOME/Downloads/Design operating systems}"
ARPO_MODEL="mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"
ARPO_REVISION=""
ARPO_DESTINATION="models/qwen-coder"
ARPO_SKIP_MODEL=0
ARPO_SKIP_SMOKE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --eco-dir) ARPO_ECO_DIR="$2"; shift 2 ;;
    --model) ARPO_MODEL="$2"; shift 2 ;;
    --revision) ARPO_REVISION="$2"; shift 2 ;;
    --destination) ARPO_DESTINATION="$2"; shift 2 ;;
    --skip-model) ARPO_SKIP_MODEL=1; shift ;;
    --skip-smoke) ARPO_SKIP_SMOKE=1; shift ;;
    --help)
      printf '%s\n' 'bash scripts/bootstrap.sh [--eco-dir DIR] [--model HF_ID] [--revision REF]' \
        '  [--destination DIR] [--skip-model] [--skip-smoke]' \
        'ARPO_PYTHON=3.13 — версия Python для uv (по умолчанию 3.13).'
      exit 0 ;;
    *) printf 'Неизвестный параметр: %s\n' "$1" >&2; exit 2 ;;
  esac
done
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  printf '%s\n' 'Автоматическая установка MLX рассчитана на Apple Silicon macOS (M1 и новее).' >&2
  exit 2
fi
ARPO_OS_VERSION="$(sw_vers -productVersion)"
if [[ "${ARPO_OS_VERSION%%.*}" -lt 14 ]]; then
  printf '%s\n' 'Для MLX нужна macOS 14 или новее. Обновите macOS перед установкой.' >&2
  exit 2
fi
if ! xcode-select -p >/dev/null 2>&1 || ! command -v clang >/dev/null 2>&1; then
  printf '%s\n' 'Сначала установите инструменты Apple: xcode-select --install' \
    'Дождитесь завершения системной установки и повторите эту команду.' >&2
  exit 2
fi
if [[ ! -d "$ARPO_ECO_DIR" ]]; then
  printf 'Папка Eco не найдена: %s\n' "$ARPO_ECO_DIR" >&2
  exit 2
fi
if [[ -z "$ARPO_REVISION" ]]; then
  if [[ "$ARPO_MODEL" == mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit ]]; then
    ARPO_REVISION="6e302ea604ad9ab206367e2c501d1571023e7b6d"
  else
    ARPO_REVISION="main"
  fi
fi
if [[ "$ARPO_SKIP_MODEL" == 0 && "$ARPO_MODEL" == mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit ]]; then
  ARPO_RAM_BYTES="$(sysctl -n hw.memsize)"
  if [[ "$ARPO_RAM_BYTES" -lt 34359738368 ]]; then
    printf '%s\n' 'Для профиля Qwen3-Coder-30B требуется больше памяти; проверенный Mac имеет 48 GB.' \
      'Для 8–24 GB выберите --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit' \
      'и --destination models/qwen-7b. Выбор модели сохраняется в отчёте.' >&2
    exit 2
  fi
fi
mkdir -p .cache .tools
export UV_CACHE_DIR="$ARPO_ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$ARPO_ROOT/.tools/python"
ARPO_UV=""
if command -v uv >/dev/null 2>&1; then
  ARPO_UV="$(command -v uv)"
elif [[ -x .tools/uv/uv ]]; then
  ARPO_UV="$ARPO_ROOT/.tools/uv/uv"
elif [[ ! -x .venv/bin/python ]]; then
  printf '%s\n' 'Загрузка uv для локальной установки Python 3.13.'
  curl --fail --location --retry 3 --silent --show-error \
    'https://astral.sh/uv/0.12.23/install.sh' --output .cache/uv-install.sh
  UV_INSTALL_DIR="$ARPO_ROOT/.tools/uv" UV_NO_MODIFY_PATH=1 sh .cache/uv-install.sh
  ARPO_UV="$ARPO_ROOT/.tools/uv/uv"
fi
if [[ ! -x .venv/bin/python ]]; then
  "$ARPO_UV" venv --python "${ARPO_PYTHON:-3.13}" .venv
fi
.venv/bin/python -c 'import sys,platform; assert sys.version_info >= (3,11); assert platform.machine()=="arm64"'
if [[ -n "$ARPO_UV" ]]; then
  "$ARPO_UV" pip install --python .venv/bin/python -r requirements-mlx-macos.txt -e .
else
  .venv/bin/python -m pip install -r requirements-mlx-macos.txt -e .
fi
ARPO_ECO_ARGS=(eco-setup --source-dir "$ARPO_ECO_DIR")
if [[ "$ARPO_SKIP_SMOKE" == 1 ]]; then ARPO_ECO_ARGS+=(--skip-smoke); fi
.venv/bin/python -m arpo_lab "${ARPO_ECO_ARGS[@]}"
if [[ "$ARPO_SKIP_MODEL" == 0 ]]; then
  if .venv/bin/python - "$ARPO_DESTINATION" "$ARPO_MODEL" "$ARPO_REVISION" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
try:
    metadata=json.loads((p/'arpo-model.json').read_text())
    if metadata['repo_id'] != sys.argv[2]:
        raise SystemExit('Destination contains another model; choose a different --destination')
    index=p/'model.safetensors.index.json'
    files=set(json.loads(index.read_text())['weight_map'].values()) if index.is_file() else \
        {f.name for f in p.glob('*.safetensors')}
    complete=bool(files) and all((p/f).is_file() and (p/f).stat().st_size>0 for f in files)
    complete=complete and (p/'config.json').is_file() and (p/'tokenizer.json').is_file()
    complete=complete and (sys.argv[3]=='main' or metadata['revision']==sys.argv[3])
    sys.exit(0 if complete else 1)
except (OSError,KeyError,ValueError):
    sys.exit(1)
PY
  then
    printf '%s\n' 'Веса выбранной модели уже скачаны; повторная загрузка не нужна.'
  else
    printf '%s\n' 'Загрузка выбранной модели. Для основной Qwen веса занимают около 17,2 GB.'
    .venv/bin/python -m arpo_lab --model "$ARPO_MODEL" download \
      --revision "$ARPO_REVISION" --destination "$ARPO_DESTINATION"
  fi
  .venv/bin/python - "$ARPO_DESTINATION" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]).resolve()
metadata=json.loads((p/'arpo-model.json').read_text())
Path('.eco/model.json').write_text(json.dumps({'path':str(p),**metadata},indent=2)+'\n')
PY
fi
printf '%s\n' 'Готово. Активация .venv не нужна. Пример настоящего компонента ACOM:' \
  'bash scripts/arpo lab-init --directory labs/abs --topic "Абсолютное значение" --name Eco.LabAbs --example abs' \
  'bash scripts/arpo lab-run labs/abs --repeats 1' \
  'Для своей темы откройте новый чат агента в этом репозитории — инструкции в AGENTS.md.'
