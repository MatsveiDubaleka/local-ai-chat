# local-ai-chat

Локальный RAG-чат на FastAPI + Ollama с поиском по документам и (опционально) по интернету. Без платных API.

## Возможности

- Чат по файлам из `~/Documents/work` (`.md`, `.txt`, `.pdf`)
- Веб-поиск через DuckDuckGo (`ddgs`) + лёгкий fetch страниц
- Русский UI: запуск/стоп Ollama, загрузка/выгрузка модели, переиндексация
- Переключатель **«Интернет»** (включён по умолчанию)

## Требования

- macOS с [Homebrew](https://brew.sh)
- Python 3.11+
- [Ollama](https://ollama.com) и модели:
  - `qwen2.5:7b`
  - `nomic-embed-text`

```bash
brew install ollama
brew services start ollama
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

## Установка и запуск

```bash
cd ~/local-ai-chat   # или клон этого репозитория
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
./start.sh
```

Откройте: **http://127.0.0.1:8787**

Остановить только веб-UI:

```bash
./stop-ui.sh
```

## Документы

1. Положите файлы в `~/Documents/work`
2. Нажмите **«Переиндексировать»** в UI
3. Спрашивайте по содержимому

Индекс хранится локально в `data/index.json` (в git не попадает).

## Интернет

Если включён чекбокс **«Интернет»**:

- вопросы про сеть / актуальность / «в интернете» → идёт веб-поиск;
- если локальные документы слабо покрывают вопрос → тоже веб;
- источники документов и URL показываются отдельно.

Поиск бесплатный (DuckDuckGo, без API-ключа). Если провайдер недоступен, в ответе будет сообщение об ошибке поиска.

## Управление памятью модели

| Кнопка | Действие |
|--------|----------|
| Запустить Ollama | `brew services start ollama` |
| Загрузить модель | короткий прогрев `qwen2.5:7b` |
| Выгрузить модель | `ollama stop` — освобождает RAM/GPU |
| Остановить Ollama | unload + `brew services stop ollama` |

## Переменные окружения

| Переменная | По умолчанию |
|------------|--------------|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` |
| `CHAT_MODEL` | `qwen2.5:7b` |
| `EMBED_MODEL` | `nomic-embed-text` |

## Структура

```
app.py              # FastAPI: RAG, веб-поиск, Ollama controls
static/index.html   # UI
start.sh / stop-ui.sh
requirements.txt
data/               # индекс (gitignore)
```
