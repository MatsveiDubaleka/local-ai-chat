#!/usr/bin/env bash
# Останавливает только веб-UI (порт 8787). Модель/Ollama не трогает.
set -euo pipefail
pids="$(lsof -tiTCP:8787 -sTCP:LISTEN 2>/dev/null || true)"
if [[ -z "${pids}" ]]; then
  echo "UI на порту 8787 не запущен"
  exit 0
fi
kill ${pids}
echo "UI остановлен (PID: ${pids})"
