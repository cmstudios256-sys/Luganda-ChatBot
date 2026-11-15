"""SQLite persistence: conversations, messages, feedback, and the export that
feeds the fine-tuning pipeline."""
from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import DATA_DIR

DB_PATH = Path(os.getenv("YOGERA_DB", str(DATA_DIR / "yogera.db")))

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT,
    language TEXT DEFAULT 'lug',
    register TEXT DEFAULT 'auto',
    teach_mode INTEGER DEFAULT 0,
    created_at TEXT,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    provider TEXT,
    model TEXT,
    latency_ms INTEGER,
    created_at TEXT,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);
CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    rating INTEGER NOT NULL,
    corrected_text TEXT,
    note TEXT,
    created_at TEXT,
    UNIQUE (message_id)
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, created_at);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_id(prefix: str = "c") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


_initialized_for: Path | None = None


def connect() -> sqlite3.Connection:
    """Open the DB, creating the schema if the file is new (or was deleted)."""
    global _initialized_for
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    if _initialized_for != DB_PATH:
        conn.executescript(SCHEMA)
        _initialized_for = DB_PATH
    return conn


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


# --------------------------------------------------------------------------- #
# Conversations & messages
# --------------------------------------------------------------------------- #
def create_conversation(
    *, language: str = "lug", register: str = "auto", teach_mode: bool = False, title: str = ""
) -> dict:
    conv_id = _new_id("c")
    now = _now()
    with connect() as conn:
        conn.execute(
            "INSERT INTO conversations (id,title,language,register,teach_mode,created_at,updated_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (conv_id, title or "Okwogera", language, register, int(teach_mode), now, now),
        )
    return {
        "id": conv_id,
        "title": title or "Okwogera",
        "language": language,
        "register": register,
        "teach_mode": teach_mode,
        "created_at": now,
    }


def touch_conversation(conv_id: str, title: str | None = None) -> None:
    with connect() as conn:
        if title:
            conn.execute(
                "UPDATE conversations SET updated_at=?, title=? WHERE id=? AND (title='Okwogera' OR title IS NULL)",
                (_now(), title[:60], conv_id),
            )
        else:
            conn.execute("UPDATE conversations SET updated_at=? WHERE id=?", (_now(), conv_id))


def add_message(
    conversation_id: str,
    role: str,
    content: str,
    *,
    provider: str = "",
    model: str = "",
    latency_ms: int = 0,
) -> dict:
    msg_id = _new_id("m")
    now = _now()
    with connect() as conn:
        conn.execute(
            "INSERT INTO messages (id,conversation_id,role,content,provider,model,latency_ms,created_at)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (msg_id, conversation_id, role, content, provider, model, latency_ms, now),
        )
    return {
        "id": msg_id,
        "conversation_id": conversation_id,
        "role": role,
        "content": content,
        "provider": provider,
        "model": model,
        "latency_ms": latency_ms,
        "created_at": now,
    }


def get_conversation(conv_id: str) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM conversations WHERE id=?", (conv_id,)).fetchone()
        if not row:
            return None
        msgs = conn.execute(
            "SELECT m.*, f.rating, f.corrected_text FROM messages m"
            " LEFT JOIN feedback f ON f.message_id = m.id"
            " WHERE m.conversation_id=? ORDER BY m.created_at, m.rowid",
            (conv_id,),
        ).fetchall()
    conv = dict(row)
    conv["messages"] = [dict(m) for m in msgs]
    return conv


def history_for_model(conv_id: str, limit: int = 20) -> list[dict[str, str]]:
    """Recent turns in the provider-neutral {role, content} shape."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE conversation_id=?"
            " AND role IN ('user','assistant') ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (conv_id, limit),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def get_message(message_id: str) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM messages WHERE id=?", (message_id,)).fetchone()
    return dict(row) if row else None


def preceding_user_message(message_id: str) -> str:
    """The user turn a given assistant message was answering."""
    with connect() as conn:
        row = conn.execute(
            "SELECT m2.content FROM messages m1 JOIN messages m2"
            "  ON m2.conversation_id = m1.conversation_id AND m2.rowid < m1.rowid"
            " WHERE m1.id = ? AND m2.role = 'user' ORDER BY m2.rowid DESC LIMIT 1",
            (message_id,),
        ).fetchone()
    return row["content"] if row else ""


def list_conversations(limit: int = 50) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT c.*, COUNT(m.id) AS turns FROM conversations c"
            " LEFT JOIN messages m ON m.conversation_id=c.id"
            " GROUP BY c.id ORDER BY c.updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_conversation(conv_id: str) -> bool:
    with connect() as conn:
        cur = conn.execute("DELETE FROM conversations WHERE id=?", (conv_id,))
        conn.execute("DELETE FROM messages WHERE conversation_id=?", (conv_id,))
        conn.execute("DELETE FROM feedback WHERE conversation_id=?", (conv_id,))
    return cur.rowcount > 0


# --------------------------------------------------------------------------- #
# Feedback
# --------------------------------------------------------------------------- #
def add_feedback(
    *, message_id: str, conversation_id: str, rating: int, corrected_text: str = "", note: str = ""
) -> dict:
    with connect() as conn:
        conn.execute(
            "INSERT INTO feedback (id,message_id,conversation_id,rating,corrected_text,note,created_at)"
            " VALUES (?,?,?,?,?,?,?)"
            " ON CONFLICT(message_id) DO UPDATE SET rating=excluded.rating,"
            " corrected_text=excluded.corrected_text, note=excluded.note",
            (_new_id("f"), message_id, conversation_id, rating, corrected_text, note, _now()),
        )
    return {"ok": True, "message_id": message_id, "rating": rating}


def stats() -> dict:
    with connect() as conn:
        convs = conn.execute("SELECT COUNT(*) c FROM conversations").fetchone()["c"]
        msgs = conn.execute("SELECT COUNT(*) c FROM messages").fetchone()["c"]
        up = conn.execute("SELECT COUNT(*) c FROM feedback WHERE rating>0").fetchone()["c"]
        down = conn.execute("SELECT COUNT(*) c FROM feedback WHERE rating<0").fetchone()["c"]
        corrections = conn.execute(
            "SELECT COUNT(*) c FROM feedback WHERE corrected_text IS NOT NULL AND corrected_text<>''"
        ).fetchone()["c"]
    return {
        "conversations": convs,
        "messages": msgs,
        "thumbs_up": up,
        "thumbs_down": down,
        "corrections": corrections,
    }


# --------------------------------------------------------------------------- #
# Export -> training data
# --------------------------------------------------------------------------- #
def _threads(include_unrated: bool = True) -> Iterable[list[sqlite3.Row]]:
    with connect() as conn:
        conv_ids = [r["id"] for r in conn.execute("SELECT id FROM conversations").fetchall()]
        for cid in conv_ids:
            rows = conn.execute(
                "SELECT m.*, f.rating, f.corrected_text FROM messages m"
                " LEFT JOIN feedback f ON f.message_id = m.id"
                " WHERE m.conversation_id=? ORDER BY m.created_at, m.rowid",
                (cid,),
            ).fetchall()
            if rows:
                yield rows


def export_sft_records(*, include_unrated: bool = True, corrections_only: bool = False) -> list[dict]:
    """Turn real chats into chat-format SFT records, dropping turns the user
    marked as bad. Human corrections are used as the target when present."""
    records: list[dict] = []
    for rows in _threads():
        messages: list[dict[str, str]] = []
        usable = True
        for row in rows:
            if row["role"] == "user":
                if messages and messages[-1]["role"] == "user":
                    messages[-1]["content"] += "\n" + row["content"]
                else:
                    messages.append({"role": "user", "content": row["content"]})
                continue
            if row["role"] != "assistant":
                continue
            rating = row["rating"]
            corrected = (row["corrected_text"] or "").strip()
            if rating is not None and rating < 0:
                if corrections_only and not corrected:
                    usable = False
                    break
                usable = False          # never teach the rejected answer
                break
            if rating is None and not include_unrated:
                usable = False
                break
            content = corrected or row["content"]
            messages.append({"role": "assistant", "content": content})
        if not usable:
            continue
        if len(messages) >= 2 and messages[0]["role"] == "user" and messages[-1]["role"] == "assistant":
            records.append({"messages": messages, "source": "yogera-app", "verified": False})
    return records


def export_preference_pairs() -> list[dict]:
    """Thumbs-down + human correction -> (prompt, chosen, rejected) for DPO."""
    pairs: list[dict] = []
    with connect() as conn:
        rows = conn.execute(
            "SELECT f.message_id, f.conversation_id, f.corrected_text, m.content"
            " FROM feedback f JOIN messages m ON m.id=f.message_id"
            " WHERE f.rating<0 AND f.corrected_text IS NOT NULL AND f.corrected_text<>''"
        ).fetchall()
        for row in rows:
            prior = conn.execute(
                "SELECT role, content FROM messages WHERE conversation_id=? AND rowid <"
                " (SELECT rowid FROM messages WHERE id=?) ORDER BY rowid DESC LIMIT 1",
                (row["conversation_id"], row["message_id"]),
            ).fetchone()
            if not prior or prior["role"] != "user":
                continue
            pairs.append(
                {
                    "prompt": prior["content"],
                    "chosen": row["corrected_text"],
                    "rejected": row["content"],
                    "source": "yogera-app",
                }
            )
    return pairs


def write_export(out_dir: Path, *, include_unrated: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    sft = export_sft_records(include_unrated=include_unrated)
    pairs = export_preference_pairs()
    sft_path = out_dir / "chats_sft.jsonl"
    pref_path = out_dir / "chats_dpo.jsonl"
    with sft_path.open("w", encoding="utf-8") as fh:
        for rec in sft:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with pref_path.open("w", encoding="utf-8") as fh:
        for rec in pairs:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {
        "sft_records": len(sft),
        "preference_pairs": len(pairs),
        "sft_path": str(sft_path),
        "dpo_path": str(pref_path),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


if __name__ == "__main__":  # `python -m app.store` prints a quick stats summary
    init_db()
    print(json.dumps(stats(), indent=2))
