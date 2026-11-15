"""Learning without a GPU: retrieval + preferred-phrasing memory.

Two things get injected into the system prompt on every turn:

  knowledge   your documents (FAQ, product facts, health content, local names),
              retrieved with SQLite FTS5 + BM25 — no vector DB, no dependencies.
  exemplars   answers that a human rated good, or rewrote by hand. Retrieved by
              similarity to the new question, so the bot reuses the phrasing a
              Muganda approved instead of inventing a new one.

This is what "learning" means when you cannot train: the model's weights do not
change, but its behaviour does, immediately, from corrections it receives. It is
also the honest first step before fine-tuning — you cannot fine-tune your way out
of not having the facts.
"""
from __future__ import annotations

import re
import sqlite3
import time
import uuid
from .store import connect

MAX_CHUNK = 700
CONTEXT_BUDGET = 3200          # characters of retrieved documents per turn
EXEMPLAR_BUDGET = 1100         # characters of preferred phrasings per turn

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    title TEXT,
    source TEXT,
    chunks INTEGER,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS exemplars (
    id TEXT PRIMARY KEY,
    conversation_id TEXT,
    user_text TEXT,
    assistant_text TEXT,
    origin TEXT,
    created_at TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
    doc_id UNINDEXED, title, body, tokenize='unicode61 remove_diacritics 0'
);
CREATE VIRTUAL TABLE IF NOT EXISTS exemplars_fts USING fts5(
    exemplar_id UNINDEXED, user_text, assistant_text UNINDEXED,
    tokenize='unicode61 remove_diacritics 0'
);
"""

def _conn() -> sqlite3.Connection:
    conn = connect()
    exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name='knowledge_fts'"
    ).fetchone()
    if not exists:
        conn.executescript(SCHEMA)
    return conn


# --------------------------------------------------------------------------- #
# Query building for FTS5 (user text must never be interpreted as FTS syntax)
# --------------------------------------------------------------------------- #
STOP = {
    "the", "and", "you", "your", "for", "are", "was", "what", "when", "where", "how",
    "can", "could", "would", "should", "please", "with", "that", "this", "have", "has",
    "ku", "mu", "ne", "ye", "kye", "ki", "ki?", "nze", "ggwe", "oli", "kale",
}


# Luganda <-> English bridge for FTS. Word-matching cannot cross languages, so a
# tiny bilingual glossary is added to every query: ask "ssente mmeka ku mmwanyi?"
# and the English FAQ that says "coffee price" still matches. Extend this with the
# vocabulary your users actually use — it is the cheapest recall win you will find.
LEXICON = {
    "ssente": ["price", "cost", "money", "fee"],
    "sente": ["price", "cost", "money"],
    "bbeeyi": ["price", "cost"],
    "ebbeeyi": ["price", "cost"],
    "muwendo": ["price", "value", "cost"],
    "mmeka": ["much", "many", "price"],
    "zingaki": ["many", "much"],
    "okugula": ["buy", "purchase", "order"],
    "okutunda": ["sell", "sale"],
    "mmwanyi": ["coffee"],
    "emwanyi": ["coffee"],
    "amazzi": ["water"],
    "emmere": ["food"],
    "mmere": ["food"],
    "ebyokulya": ["food"],
    "essimu": ["phone", "mobile", "call"],
    "ssimu": ["phone", "mobile", "call"],
    "okutuuka": ["delivery", "arrive", "reach"],
    "okuleeta": ["delivery", "bring"],
    "okutwala": ["delivery", "take"],
    "olunaku": ["day"],
    "ennaku": ["days"],
    "lunaku": ["day"],
    "ekiseera": ["time", "hours"],
    "kiseera": ["time", "hours"],
    "obudde": ["time", "weather"],
    "okudda": ["return", "refund"],
    "okuzza": ["return", "refund"],
    "ssomero": ["school"],
    "ennyumba": ["house", "home"],
    "obulwadde": ["health", "sick", "illness"],
    "eddwaaliro": ["hospital", "clinic"],
    "omusawo": ["doctor", "clinician"],
    "eddagala": ["medicine", "drug"],
    "obuyambi": ["help", "support"],
    "okuyamba": ["help", "support"],
    "omulimu": ["job", "work"],
    "emirimu": ["jobs", "work"],
    "okukola": ["work", "do"],
    "abaana": ["children"],
    "omukulu": ["manager", "leader", "adult"],
    "akawunti": ["account"],
    "bbanka": ["bank"],
    "banka": ["bank"],
    "akasaale": ["loan", "credit"],
    "okusasula": ["pay", "payment"],
    "okutereka": ["save", "savings"],
}
ENGLISH_TO_LUGANDA: dict[str, list[str]] = {}
for lug_word, english in LEXICON.items():
    for eng in english:
        ENGLISH_TO_LUGANDA.setdefault(eng, []).append(lug_word)


def _match_expr(query: str, *, max_terms: int = 18) -> str:
    tokens = re.findall(r"[0-9a-zA-Z']{2,}", query.lower())
    # Emit per token so one long word cannot starve the rest of the query:
    # the word itself, a 5-char prefix (Luganda prefixes change: essomero/ssomero),
    # and up to two glossary neighbours across the language gap.
    terms: list[str] = []
    for tok in tokens:
        if tok in STOP:
            continue
        terms.append(f'"{tok}"')
        if len(tok) >= 6:
            terms.append(f'"{tok[:5]}"*')
        neighbours = (LEXICON.get(tok, []) + ENGLISH_TO_LUGANDA.get(tok, []))[:2]
        for mapped in neighbours:
            terms.append(f'"{mapped}"')
        if len(terms) >= max_terms:
            break
    return " OR ".join(dict.fromkeys(terms))


def _chunks(text: str, size: int = MAX_CHUNK) -> list[str]:
    text = re.sub(r"\r\n?", "\n", text).strip()
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out: list[str] = []
    for para in paragraphs:
        if len(para) <= size:
            out.append(para)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", para)
        buf = ""
        for sentence in sentences:
            if len(buf) + len(sentence) + 1 > size and buf:
                out.append(buf.strip())
                buf = sentence
            else:
                buf = f"{buf} {sentence}".strip()
        if buf:
            out.append(buf.strip())
    return out or ([text] if text else [])


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #
def add_document(title: str, text: str, source: str = "manual") -> dict:
    doc_id = f"d_{uuid.uuid4().hex[:10]}"
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    pieces = _chunks(text)
    with _conn() as conn:
        conn.execute(
            "INSERT INTO documents (id,title,source,chunks,created_at) VALUES (?,?,?,?,?)",
            (doc_id, title.strip()[:200] or "Untitled", source, len(pieces), now),
        )
        for piece in pieces:
            conn.execute(
                "INSERT INTO knowledge_fts (doc_id,title,body) VALUES (?,?,?)",
                (doc_id, title, piece),
            )
    return {"id": doc_id, "title": title, "chunks": len(pieces), "created_at": now}


def list_documents(limit: int = 100) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM documents ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def delete_document(doc_id: str) -> bool:
    with _conn() as conn:
        cur = conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        conn.execute("DELETE FROM knowledge_fts WHERE doc_id=?", (doc_id,))
    return cur.rowcount > 0


def search_documents(query: str, k: int = 4, budget: int = CONTEXT_BUDGET) -> list[dict]:
    expr = _match_expr(query)
    if not expr:
        return []
    with _conn() as conn:
        try:
            rows = conn.execute(
                "SELECT doc_id, title, snippet(knowledge_fts, 2, '', '', '…', 24) AS body,"
                "       bm25(knowledge_fts) AS rank"
                " FROM knowledge_fts WHERE knowledge_fts MATCH ? ORDER BY rank LIMIT ?",
                (expr, k),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
    out, used, seen = [], 0, set()
    for row in rows:
        body = (row["body"] or "").strip()
        if not body or body in seen:
            continue
        if used + len(body) > budget and out:
            break
        seen.add(body)
        used += len(body)
        out.append({"doc_id": row["doc_id"], "title": row["title"], "text": body,
                    "score": round(-row["rank"], 3)})
    return out


# --------------------------------------------------------------------------- #
# Exemplars: gold answers from humans
# --------------------------------------------------------------------------- #
def add_exemplar(user_text: str, assistant_text: str, *, conversation_id: str = "",
                 origin: str = "feedback") -> dict | None:
    user_text, assistant_text = (user_text or "").strip(), (assistant_text or "").strip()
    if len(user_text) < 2 or len(assistant_text) < 2:
        return None
    ex_id = f"x_{uuid.uuid4().hex[:10]}"
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with _conn() as conn:
        conn.execute(
            "INSERT INTO exemplars (id,conversation_id,user_text,assistant_text,origin,created_at)"
            " VALUES (?,?,?,?,?,?)",
            (ex_id, conversation_id, user_text, assistant_text, origin, now),
        )
        conn.execute(
            "INSERT INTO exemplars_fts (exemplar_id,user_text,assistant_text) VALUES (?,?,?)",
            (ex_id, user_text, assistant_text),
        )
    return {"id": ex_id, "user_text": user_text[:120], "assistant_text": assistant_text[:200]}


def search_exemplars(query: str, k: int = 3, budget: int = EXEMPLAR_BUDGET) -> list[dict]:
    expr = _match_expr(query)
    if not expr:
        return []
    with _conn() as conn:
        try:
            rows = conn.execute(
                "SELECT exemplar_id, user_text, assistant_text, bm25(exemplars_fts) AS rank"
                " FROM exemplars_fts WHERE exemplars_fts MATCH ? ORDER BY rank LIMIT ?",
                (expr, k),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
    out, used = [], 0
    for row in rows:
        if used + len(row["assistant_text"]) > budget and out:
            break
        used += len(row["assistant_text"])
        out.append({"user": row["user_text"], "assistant": row["assistant_text"],
                    "score": round(-row["rank"], 3)})
    return out


# --------------------------------------------------------------------------- #
# Prompt assembly
# --------------------------------------------------------------------------- #
def build_context(query: str, *, docs: int = 4, exemplars: int = 3) -> dict:
    documents = search_documents(query, k=docs)
    gold = search_exemplars(query, k=exemplars)
    parts: list[str] = []
    if documents:
        parts.append("FACTS you may rely on (from the operator's own documents; use them only when relevant,"
                     " never contradict them, and never mention that you were handed them):")
        for doc in documents:
            parts.append(f"- [{doc['title']}] {doc['text']}")
    if gold:
        parts.append("PREFERRED PHRASING — answers a Luganda speaker approved for similar questions."
                     " Match the register and the vocabulary; do not copy them verbatim if the question differs:")
        for pair in gold:
            parts.append(f"- Q: {pair['user']}\n  A: {pair['assistant']}")
    return {
        "documents": documents,
        "exemplars": gold,
        "prompt_block": "\n".join(parts),
        "chars": sum(len(p) for p in parts),
    }


def stats() -> dict:
    with _conn() as conn:
        docs = conn.execute("SELECT COUNT(*) c FROM documents").fetchone()["c"]
        chunks = conn.execute("SELECT COUNT(*) c FROM knowledge_fts").fetchone()["c"]
        ex = conn.execute("SELECT COUNT(*) c FROM exemplars").fetchone()["c"]
    return {"documents": docs, "chunks": chunks, "exemplars": ex}
