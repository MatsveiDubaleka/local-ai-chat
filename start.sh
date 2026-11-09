#!/usr/bin/env bash
# Запуск веб-UI: http://127.0.0.1:8787
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
fi

# Ensure Ollama is up (does not load the LLM until first chat / «Загрузить модель»)
if ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  brew services start ollama || true
  for _ in $(seq 1 25); do
    curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && break
    sleep 0.4
  done
fi

export CHAT_MODEL="${CHAT_MODEL:-qwen2.5:7b}"
export EMBED_MODEL="${EMBED_MODEL:-nomic-embed-text}"
export OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"

exec .venv/bin/uvicorn app:app --host 127.0.0.1 --port 8787
