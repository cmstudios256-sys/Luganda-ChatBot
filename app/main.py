"""Yogera — Luganda chatbot API + web UI.

Run:  uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import store
from .config import ROOT, SETTINGS, WEB_DIR
from .prompts import build_system_prompt
from .providers import ProviderError, get_provider, provider_status

app = FastAPI(title="Yogera — Luganda Chatbot", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

store.init_db()


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    conversation_id: str | None = None
    language: str = "lug"          # lug | en
    tone: str = "auto"             # auto | peer | respect
    teach_mode: bool = False
    english_gloss: bool = False
    temperature: float | None = None


class FeedbackRequest(BaseModel):
    message_id: str
    conversation_id: str
    rating: int                   # 1 or -1
    corrected_text: str = ""
    note: str = ""


class SettingsRequest(BaseModel):
    provider: str | None = None
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    temperature: float | None = None


# --------------------------------------------------------------------------- #
# Meta
# --------------------------------------------------------------------------- #
@app.get("/api/health")
async def health():
    return {"ok": True, "status": provider_status(), "stats": store.stats()}


@app.get("/api/settings")
async def read_settings():
    return provider_status()


@app.post("/api/settings")
async def update_settings(payload: SettingsRequest):
    """Runtime override so you can test a key without restarting the server.
    Nothing is written to disk — put permanent keys in .env."""
    if payload.provider:
        SETTINGS.provider = payload.provider.lower()
    if payload.model is not None:
        SETTINGS.model = payload.model.strip()
        if SETTINGS.model:
            os.environ["LLM_MODEL"] = SETTINGS.model
    if payload.base_url is not None:
        SETTINGS.base_url = payload.base_url.strip()
        if SETTINGS.base_url:
            os.environ["LLM_BASE_URL"] = SETTINGS.base_url
    if payload.api_key:
        key = payload.api_key.strip()
        env_name = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "compatible": "LLM_API_KEY",
        }.get(SETTINGS.resolved_provider)
        if env_name:
            os.environ[env_name] = key
    if payload.temperature is not None:
        SETTINGS.temperature = max(0.0, min(1.5, payload.temperature))
    return provider_status()


# --------------------------------------------------------------------------- #
# Chat (SSE)
# --------------------------------------------------------------------------- #
def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request):
    conv_id = req.conversation_id
    if conv_id:
        if store.get_conversation(conv_id) is None:
            raise HTTPException(404, "conversation not found")
    else:
        conv = store.create_conversation(
            language=req.language, register=req.tone, teach_mode=req.teach_mode
        )
        conv_id = conv["id"]

    user_msg = store.add_message(conv_id, "user", req.message.strip())
    store.touch_conversation(conv_id, title=req.message.strip()[:60])

    system = build_system_prompt(
        language=req.language,
        register=req.tone,
        teach_mode=req.teach_mode,
        english_gloss=req.english_gloss,
    )
    history = store.history_for_model(conv_id)  # already ends with this message
    provider = get_provider()
    status = provider_status()

    async def event_stream() -> AsyncIterator[str]:
        yield _sse(
            "meta",
            {
                "conversation_id": conv_id,
                "user_message_id": user_msg["id"],
                **status,
            },
        )
        started = time.perf_counter()
        parts: list[str] = []
        try:
            async for piece in provider.stream(
                history,
                system=system,
                temperature=req.temperature if req.temperature is not None else SETTINGS.temperature,
                max_tokens=SETTINGS.max_tokens,
            ):
                parts.append(piece)
                yield _sse("delta", {"text": piece})
                if await request.is_disconnected():
                    break
        except ProviderError as exc:
            yield _sse("error", {"message": str(exc), "kind": "provider"})
        except Exception as exc:  # network, DNS, timeout...
            yield _sse("error", {"message": f"{type(exc).__name__}: {exc}", "kind": "unknown"})

        text = "".join(parts).strip()
        if not text and not parts:
            text = (
                "Nkusaba ondinde — waaliwo obuzibu mu kutuuka ku kyuma ekitegeera. "
                "Kebera API key yo oba internet, oluvannyuma ogezeeko nate."
            )
        latency = int((time.perf_counter() - started) * 1000)
        saved = store.add_message(
            conv_id,
            "assistant",
            text,
            provider=status["provider"],
            model=status["model"],
            latency_ms=latency,
        )
        yield _sse("done", {"message": saved, "latency_ms": latency})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@app.post("/api/chat/sync")
async def chat_sync(req: ChatRequest):
    """Non-streaming convenience endpoint (same behaviour, one JSON reply)."""
    if req.conversation_id and store.get_conversation(req.conversation_id) is None:
        raise HTTPException(404, "conversation not found")
    conv_id = req.conversation_id or store.create_conversation(
        language=req.language, register=req.tone, teach_mode=req.teach_mode
    )["id"]
    store.add_message(conv_id, "user", req.message.strip())
    store.touch_conversation(conv_id, title=req.message.strip()[:60])
    provider = get_provider()
    started = time.perf_counter()
    try:
        text = await provider.complete(
            store.history_for_model(conv_id),
            system=build_system_prompt(
                language=req.language,
                register=req.tone,
                teach_mode=req.teach_mode,
                english_gloss=req.english_gloss,
            ),
            temperature=req.temperature if req.temperature is not None else SETTINGS.temperature,
            max_tokens=SETTINGS.max_tokens,
        )
    except (ProviderError, Exception) as exc:  # noqa: BLE001
        raise HTTPException(502, f"provider error: {exc}")
    latency = int((time.perf_counter() - started) * 1000)
    saved = store.add_message(
        conv_id, "assistant", text.strip(), provider=provider.name, model=provider.model, latency_ms=latency
    )
    return {
        "conversation_id": conv_id,
        "reply": saved["content"],
        "message_id": saved["id"],
        "latency_ms": latency,
        "provider": provider.name,
        "model": provider.model,
        "demo_mode": not SETTINGS.is_live,
    }


# --------------------------------------------------------------------------- #
# Conversations & feedback
# --------------------------------------------------------------------------- #
@app.get("/api/conversations")
async def conversations():
    return {"items": store.list_conversations()}


@app.get("/api/conversations/{conv_id}")
async def conversation(conv_id: str):
    conv = store.get_conversation(conv_id)
    if not conv:
        raise HTTPException(404, "conversation not found")
    return conv


@app.delete("/api/conversations/{conv_id}")
async def remove_conversation(conv_id: str):
    return {"deleted": store.delete_conversation(conv_id)}


@app.post("/api/feedback")
async def feedback(req: FeedbackRequest):
    if req.rating not in (1, -1):
        raise HTTPException(400, "rating must be 1 or -1")
    return store.add_feedback(
        message_id=req.message_id,
        conversation_id=req.conversation_id,
        rating=req.rating,
        corrected_text=req.corrected_text.strip(),
        note=req.note.strip(),
    )


# --------------------------------------------------------------------------- #
# Learning loop -> training data
# --------------------------------------------------------------------------- #
@app.post("/api/export")
async def export(include_unrated: bool = True):
    out_dir = ROOT / "finetune" / "data" / "exported"
    return JSONResponse(store.write_export(out_dir, include_unrated=include_unrated))


@app.get("/api/export/download")
async def export_download():
    out_dir = ROOT / "finetune" / "data" / "exported"
    result = store.write_export(out_dir)
    path = Path(result["sft_path"])
    if not path.exists() or result["sft_records"] == 0:
        raise HTTPException(404, "nothing to export yet — chat a little first")
    return FileResponse(path, filename="yogera_chats_sft.jsonl", media_type="application/jsonl")


# --------------------------------------------------------------------------- #
# Static UI
# --------------------------------------------------------------------------- #
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(WEB_DIR / "index.html")
