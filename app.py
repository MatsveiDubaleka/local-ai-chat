#!/usr/bin/env python3
"""Локальный чат с RAG по ~/Documents/work + веб-поиск + управление Ollama."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pypdf import PdfReader

APP_DIR = Path(__file__).resolve().parent
DOCS_DIR = Path(os.path.expanduser("~/Documents/work")).resolve()
INDEX_PATH = APP_DIR / "data" / "index.json"
STATIC_DIR = APP_DIR / "static"

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "qwen2.5:7b")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 4
WEB_RESULTS = 4
WEB_FETCH_CHARS = 1800
TEXT_EXTS = {".md", ".txt", ".pdf"}
MAX_FILE_BYTES = 20 * 1024 * 1024  # skip huge files

# Russian / web-intent cues (lowercase match)
WEB_INTENT_RE = re.compile(
    r"("
    r"интернет|в\s+сети|онлайн|в\s+веб|website|web\b|google|гугл|"
    r"поиск(ай|ать)?\s+в|найди\s+в\s+интернет|загруз(ить|ка)\s+из\s+интернет|"
    r"актуальн|свеж(ие|ая|ий)|сегодня|сейчас|текущ|новост|погода|"
    r"курс\s+валют|bitcoin|btc|кто\s+такой|что\s+такое|"
    r"can you (search|browse|look up)|from the (internet|web)"
    r")",
    re.IGNORECASE,
)

app = FastAPI(title="Локальный ИИ-чат с документами")
STATIC_DIR.mkdir(parents=True, exist_ok=True)
(APP_DIR / "data").mkdir(parents=True, exist_ok=True)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    unload_after: bool = False
    use_web: bool = True


class ActionResponse(BaseModel):
    ok: bool
    message: str
    status: dict[str, Any]


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _chunk_text(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + CHUNK_SIZE)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - CHUNK_OVERLAP)
    return chunks


def _read_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        parts: list[str] = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return "\n".join(parts)
    return ""


def _ollama_reachable() -> bool:
    try:
        with httpx.Client(timeout=2.0) as client:
            r = client.get(f"{OLLAMA_HOST}/api/tags")
            return r.status_code == 200
    except Exception:
        return False


def _running_models() -> list[dict[str, Any]]:
    try:
        with httpx.Client(timeout=3.0) as client:
            r = client.get(f"{OLLAMA_HOST}/api/ps")
            r.raise_for_status()
            return r.json().get("models") or []
    except Exception:
        return []


def _brew_ollama_status() -> str:
    try:
        out = subprocess.run(
            ["brew", "services", "list"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        for line in out.stdout.splitlines():
            if line.startswith("ollama"):
                parts = line.split()
                return parts[1] if len(parts) > 1 else "unknown"
    except Exception:
        pass
    return "unknown"


def get_status() -> dict[str, Any]:
    reachable = _ollama_reachable()
    running = _running_models() if reachable else []
    names = [m.get("name") or m.get("model") or "" for m in running]
    chat_loaded = any(CHAT_MODEL in n for n in names)
    embed_loaded = any(EMBED_MODEL in n for n in names)
    index = _load_index()
    return {
        "ollama_reachable": reachable,
        "brew_service": _brew_ollama_status(),
        "chat_model": CHAT_MODEL,
        "embed_model": EMBED_MODEL,
        "chat_loaded": chat_loaded,
        "embed_loaded": embed_loaded,
        "loaded_models": names,
        "docs_dir": str(DOCS_DIR),
        "index_chunks": len(index.get("chunks") or []),
        "index_files": len(index.get("files") or {}),
        "indexed_at": index.get("indexed_at"),
        "web_search": True,
    }


def _embed(texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    with httpx.Client(timeout=120.0) as client:
        for text in texts:
            r = client.post(
                f"{OLLAMA_HOST}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": text},
            )
            if r.status_code >= 400:
                raise HTTPException(
                    status_code=502,
                    detail=f"Ошибка эмбеддинга ({r.status_code}): {r.text[:300]}",
                )
            vectors.append(r.json()["embedding"])
    return vectors


def _load_index() -> dict[str, Any]:
    if not INDEX_PATH.exists():
        return {"chunks": [], "files": {}, "indexed_at": None}
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


def _save_index(data: dict[str, Any]) -> None:
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def reindex() -> dict[str, Any]:
    if not _ollama_reachable():
        raise HTTPException(status_code=503, detail="Ollama не запущена. Нажмите «Запустить Ollama».")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    old = _load_index()
    old_files: dict[str, Any] = old.get("files") or {}
    new_files: dict[str, Any] = {}
    chunks: list[dict[str, Any]] = []
    skipped: list[str] = []
    reused = 0

    paths = sorted(
        p
        for p in DOCS_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in TEXT_EXTS and not p.name.startswith(".")
    )

    for path in paths:
        try:
            size = path.stat().st_size
            if size > MAX_FILE_BYTES:
                skipped.append(f"{path.name} (слишком большой: {size} байт)")
                continue
            raw = _read_file(path)
            digest = hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()
            rel = str(path.relative_to(DOCS_DIR))
            prev = old_files.get(rel)
            if prev and prev.get("sha256") == digest:
                for ch in old.get("chunks") or []:
                    if ch.get("file") == rel:
                        chunks.append(ch)
                        reused += 1
                new_files[rel] = prev
                continue

            file_chunks = _chunk_text(raw)
            if not file_chunks:
                new_files[rel] = {"sha256": digest, "chunks": 0}
                continue
            vectors = _embed(file_chunks)
            for i, (text, vec) in enumerate(zip(file_chunks, vectors)):
                chunks.append(
                    {
                        "id": f"{rel}::{i}",
                        "file": rel,
                        "text": text,
                        "embedding": vec,
                    }
                )
            new_files[rel] = {"sha256": digest, "chunks": len(file_chunks)}
        except Exception as exc:  # noqa: BLE001 — collect per-file errors for UI
            skipped.append(f"{path.name}: {exc}")

    data = {
        "chunks": chunks,
        "files": new_files,
        "indexed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "skipped": skipped,
        "reused_chunks": reused,
    }
    _save_index(data)
    return {
        "files": len(new_files),
        "chunks": len(chunks),
        "reused_chunks": reused,
        "skipped": skipped,
        "indexed_at": data["indexed_at"],
    }


def retrieve(query: str, k: int = TOP_K) -> list[dict[str, Any]]:
    index = _load_index()
    chunks = index.get("chunks") or []
    if not chunks:
        return []
    qvec = _embed([query])[0]
    scored = []
    for ch in chunks:
        score = _cosine(qvec, ch.get("embedding") or [])
        scored.append({**ch, "score": score})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return [
        {"file": c["file"], "text": c["text"], "score": round(c["score"], 4)}
        for c in scored[:k]
        if c["score"] > 0.15
    ]


def _wants_web(message: str, use_web: bool) -> bool:
    if not use_web:
        return False
    return bool(WEB_INTENT_RE.search(message))


def _should_also_web_for_weak_docs(message: str, hits: list[dict[str, Any]], use_web: bool) -> bool:
    """If toggle is on and local docs don't cover the question well, fall back to web."""
    if not use_web:
        return False
    if _wants_web(message, True):
        return True
    if not hits:
        return True
    best = hits[0].get("score") or 0.0
    return best < 0.42


def _fetch_page_text(url: str) -> str:
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "ru,en;q=0.8",
        }
        with httpx.Client(timeout=12.0, follow_redirects=True, headers=headers) as client:
            r = client.get(url)
            if r.status_code >= 400:
                return ""
            ctype = (r.headers.get("content-type") or "").lower()
            if "html" not in ctype and "text" not in ctype:
                return ""
            soup = BeautifulSoup(r.text, "lxml")
            for tag in soup(["script", "style", "nav", "footer", "noscript", "aside"]):
                tag.decompose()
            text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
            return text[:WEB_FETCH_CHARS]
    except Exception:
        return ""


def _ddgs_text(query: str, max_results: int) -> list[dict[str, Any]]:
    """Try a few keyless backends; retry once on empty/transient errors."""
    backends: list[str | None] = [None, "duckduckgo", "bing"]
    last_err: Exception | None = None
    for attempt in range(2):
        for backend in backends:
            try:
                kwargs: dict[str, Any] = {
                    "region": "ru-ru",
                    "max_results": max_results,
                }
                if backend:
                    kwargs["backend"] = backend
                with DDGS() as ddgs:
                    raw = list(ddgs.text(query, **kwargs))
                if raw:
                    return raw
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                continue
        if attempt == 0:
            time.sleep(0.6)
    if last_err:
        raise last_err
    return []


def web_search(query: str, max_results: int = WEB_RESULTS) -> list[dict[str, Any]]:
    """Keyless DuckDuckGo/Bing search via ddgs; optionally fetch page snippets."""
    results: list[dict[str, Any]] = []
    try:
        raw = _ddgs_text(query, max_results)
    except Exception as exc:  # noqa: BLE001
        return [
            {
                "title": "Ошибка поиска",
                "url": "",
                "host": "",
                "snippet": f"Веб-поиск недоступен: {exc}",
                "text": "",
            }
        ]

    if not raw:
        return [
            {
                "title": "Поиск без результатов",
                "url": "",
                "host": "",
                "snippet": "Поисковик не вернул ссылок. Можно переформулировать вопрос.",
                "text": "",
            }
        ]

    for item in raw:
        url = (item.get("href") or item.get("url") or "").strip()
        title = (item.get("title") or "").strip() or url
        snippet = (item.get("body") or item.get("snippet") or "").strip()
        page_text = _fetch_page_text(url) if url else ""
        host = urlparse(url).netloc if url else ""
        results.append(
            {
                "title": title,
                "url": url,
                "host": host,
                "snippet": snippet,
                "text": page_text or snippet,
            }
        )
    return results


def _chat_completion(prompt: str, keep_alive: str | int = "5m") -> str:
    with httpx.Client(timeout=180.0) as client:
        r = client.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": CHAT_MODEL,
                "prompt": prompt,
                "stream": False,
                "keep_alive": keep_alive,
                "options": {
                    "temperature": 0.3,
                },
            },
        )
        if r.status_code >= 400:
            raise HTTPException(
                status_code=502,
                detail=f"Ошибка генерации ({r.status_code}): {r.text[:400]}",
            )
        return (r.json().get("response") or "").strip()


def _strip_cjk_artifacts(text: str) -> str:
    """Remove accidental Chinese/CJK translation dumps common with some Qwen outputs."""
    if not text:
        return text
    # Drop lines that are mostly CJK characters
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        cjk = len(re.findall(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", line))
        if cjk and cjk / max(len(line), 1) > 0.25:
            continue
        cleaned_lines.append(line)
    out = "\n".join(cleaned_lines).strip()
    # Also strip inline CJK runs longer than a few chars
    out = re.sub(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]{3,}", "", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out.strip() or text


def unload_models(models: list[str] | None = None) -> list[str]:
    """Unload models from RAM/VRAM. Prefers `ollama stop`, falls back to keep_alive=0."""
    targets = models or [CHAT_MODEL, EMBED_MODEL]
    unloaded: list[str] = []
    running = {m.get("name") or m.get("model") or "" for m in _running_models()}

    for model in targets:
        if not any(model in name for name in running) and model not in running:
            pass
        try:
            proc = subprocess.run(
                ["ollama", "stop", model],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if proc.returncode == 0:
                unloaded.append(model)
                continue
        except Exception:
            pass
        try:
            with httpx.Client(timeout=60.0) as client:
                client.post(
                    f"{OLLAMA_HOST}/api/generate",
                    json={
                        "model": model,
                        "prompt": ".",
                        "stream": False,
                        "keep_alive": 0,
                    },
                )
            unloaded.append(model)
        except Exception:
            pass
    return unloaded


def start_ollama() -> str:
    if _ollama_reachable():
        return "Ollama уже запущена"
    proc = subprocess.run(
        ["brew", "services", "start", "ollama"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    for _ in range(20):
        time.sleep(0.4)
        if _ollama_reachable():
            return (proc.stdout or proc.stderr or "Ollama запущена").strip()
    raise HTTPException(
        status_code=503,
        detail=f"Не удалось дождаться API Ollama. stdout={proc.stdout!r} stderr={proc.stderr!r}",
    )


def stop_ollama_service() -> str:
    unload_models()
    proc = subprocess.run(
        ["brew", "services", "stop", "ollama"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    for _ in range(15):
        time.sleep(0.3)
        if not _ollama_reachable():
            break
    return (proc.stdout or proc.stderr or "Ollama остановлена").strip()


def _build_prompt(
    message: str,
    doc_hits: list[dict[str, Any]],
    web_hits: list[dict[str, Any]],
    used_web: bool,
) -> str:
    doc_blocks = []
    for i, h in enumerate(doc_hits, 1):
        doc_blocks.append(f"[Д{i}] Файл: {h['file']}\n{h['text']}")
    docs_ctx = "\n\n".join(doc_blocks) if doc_blocks else "(локальные документы не подошли)"

    web_blocks = []
    for i, h in enumerate(web_hits, 1):
        body = h.get("text") or h.get("snippet") or ""
        web_blocks.append(
            f"[В{i}] {h.get('title') or 'Источник'}\nURL: {h.get('url') or '—'}\n{body}"
        )
    web_ctx = "\n\n".join(web_blocks) if web_blocks else "(веб-результаты пусты)"

    if used_web:
        mode_note = (
            "У тебя ЕСТЬ доступ к интернету через встроенный веб-поиск этого приложения. "
            "Если спрашивают, можешь ли ты грузить/искать в интернете — ответь ДА: "
            "поиск уже выполнен (см. блок ВЕБ), цитируй URL. "
            "Не говори, что ты offline-only или docs-only. "
            "Для вопросов про локальные файлы опирайся на ДОКУМЕНТЫ; для сети — на ВЕБ. "
            "Можно комбинировать."
        )
    else:
        mode_note = (
            "Интернет выключен переключателем. Отвечай только по локальным документам; "
            "если данных нет — честно скажи, что в документах этого нет, "
            "и предложи включить «Интернет»."
        )

    return (
        "Ты локальный помощник на этом Mac.\n"
        "ЯЗЫК: отвечай ТОЛЬКО на языке пользователя. Сейчас пользователь пишет по-русски — "
        "весь ответ должен быть по-русски. ЗАПРЕЩЕНО писать иероглифы, китайский, японский "
        "или любой перевод на другие языки. Никаких bilingual dump.\n"
        f"{mode_note}\n"
        "Кратко и точно. Указывай номера источников [Д1]/[В1] где уместно.\n\n"
        f"ДОКУМЕНТЫ:\n{docs_ctx}\n\n"
        f"ВЕБ:\n{web_ctx if used_web else '(не использовался)'}\n\n"
        f"ВОПРОС: {message}\n\n"
        "ОТВЕТ (только русский):"
    )


@app.get("/", response_class=HTMLResponse)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def api_status() -> dict[str, Any]:
    return get_status()


@app.post("/api/start", response_model=ActionResponse)
def api_start() -> ActionResponse:
    msg = start_ollama()
    return ActionResponse(ok=True, message=msg, status=get_status())


@app.post("/api/stop-service", response_model=ActionResponse)
def api_stop_service() -> ActionResponse:
    msg = stop_ollama_service()
    return ActionResponse(ok=True, message=msg, status=get_status())


@app.post("/api/unload", response_model=ActionResponse)
def api_unload() -> ActionResponse:
    if not _ollama_reachable():
        return ActionResponse(
            ok=True,
            message="Ollama уже остановлена — модели не в памяти",
            status=get_status(),
        )
    unloaded = unload_models()
    still = _running_models()
    if still:
        names = [m.get("name") or "?" for m in still]
        msg = f"Выгружено: {', '.join(unloaded) or '—'}. Ещё в памяти: {', '.join(names)}"
    else:
        msg = (
            f"Модели выгружены из памяти ({', '.join(unloaded) or 'не было загружено'}). "
            "Демон Ollama работает, но RAM/GPU модели свободны."
        )
    return ActionResponse(ok=True, message=msg, status=get_status())


@app.post("/api/warm", response_model=ActionResponse)
def api_warm() -> ActionResponse:
    if not _ollama_reachable():
        start_ollama()
    _chat_completion("Ответь одним словом: ок", keep_alive="5m")
    return ActionResponse(ok=True, message=f"Модель {CHAT_MODEL} загружена", status=get_status())


@app.post("/api/reindex")
def api_reindex() -> dict[str, Any]:
    result = reindex()
    return {"ok": True, "result": result, "status": get_status()}


@app.post("/api/chat")
def api_chat(body: ChatRequest) -> dict[str, Any]:
    if not _ollama_reachable():
        raise HTTPException(status_code=503, detail="Ollama не запущена. Нажмите «Запустить Ollama».")

    index = _load_index()
    if not (index.get("chunks") or []):
        reindex()

    hits = retrieve(body.message)
    used_web = _should_also_web_for_weak_docs(body.message, hits, body.use_web)
    web_hits: list[dict[str, Any]] = []
    if used_web:
        web_hits = web_search(body.message)

    prompt = _build_prompt(body.message, hits, web_hits, used_web)
    keep_alive: str | int = 0 if body.unload_after else "5m"
    answer = _strip_cjk_artifacts(_chat_completion(prompt, keep_alive=keep_alive))
    if body.unload_after:
        unload_models()

    web_sources = [
        {
            "title": w.get("title") or "",
            "url": w.get("url") or "",
            "host": w.get("host") or "",
            "snippet": (w.get("snippet") or "")[:240],
        }
        for w in web_hits
        if w.get("url") or w.get("snippet")
    ]
    return {
        "ok": True,
        "answer": answer,
        "sources": hits,
        "web_sources": web_sources,
        "used_web": used_web,
        "status": get_status(),
    }


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8787, reload=False)
