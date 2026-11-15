"""Shared plumbing for the corpus ingest scripts: filtering, scoring, writing."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from .langid import classify
from .text_extract import Segment

OUT = Path(__file__).resolve().parent / "out"
SEGMENTS = OUT / "segments"


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    # PDFs and OCR hard-wrap lines mid-sentence. Unwrap before anything else, or every
    # downstream heuristic (dedupe, chunking, boilerplate) sees line fragments.
    text = re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", text)          # hyphenated breaks
    text = re.sub(r"(?<=[a-zA-Z,;:])\n(?=[a-z])", " ", text)    # plain line breaks
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def content_hash(text: str) -> str:
    key = re.sub(r"[^a-z0-9 ]", "", normalize(text).lower())
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def process(
    segments: list[Segment],
    *,
    min_words: int = 25,
    keep_english: bool = False,
    keep_mixed: bool = True,
    min_luganda_score: float = 0.05,
) -> tuple[list[dict], dict]:
    """Filter a batch of segments and attach language provenance.

    Rejections are counted by reason — that number is the honest measure of how much
    of a "big" source was actually usable Luganda.
    """
    kept: list[dict] = []
    stats: Counter = Counter()
    seen: set[str] = set()

    for seg in segments:
        stats["segments_in"] += 1
        text = normalize(seg.text)
        words = len(text.split())

        if seg.meta.get("needs_ocr"):
            stats["rejected_needs_ocr"] += 1
            continue
        if words < min_words:
            stats["rejected_too_short"] += 1
            continue

        lang = classify(text)
        if lang.decision == "eng" and not keep_english:
            stats["rejected_english"] += 1
            continue
        if lang.decision in {"unknown", "other"}:
            stats["rejected_not_luganda"] += 1
            continue
        if lang.decision == "mixed" and not keep_mixed:
            stats["rejected_code_mix"] += 1
            continue
        if lang.lug_score < min_luganda_score and lang.decision != "mixed":
            stats["rejected_low_luganda"] += 1
            continue

        digest = content_hash(text)
        if digest in seen:
            stats["rejected_duplicate"] += 1
            continue
        seen.add(digest)

        kept.append({
            "id": digest,
            "text": text,
            "source": seg.source,
            "kind": seg.kind,
            "locator": seg.locator,
            "words": words,
            "lang": lang.as_dict(),
            # Everything from an outside source is unverified until a native speaker
            # signs off. Keep the flag on every record; the trainer can filter on it.
            "verified": bool(seg.meta.get("verified", False)),
            "modality": seg.meta.get("modality", "text"),
            **{k: v for k, v in seg.meta.items() if k not in {"verified", "modality", "needs_ocr"}},
        })
        stats["kept"] += 1
        stats[f"kept_words_{seg.kind}"] += words
        if lang.code_mix:
            stats["kept_code_mixed"] += 1
        if seg.meta.get("modality") == "spoken":
            stats["kept_spoken"] += 1

    stats["kept_words"] = sum(r["words"] for r in kept)
    return kept, dict(stats)


def write_segments(name: str, records: list[dict]) -> Path:
    SEGMENTS.mkdir(parents=True, exist_ok=True)
    path = SEGMENTS / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def read_all_segments() -> list[dict]:
    records: list[dict] = []
    if not SEGMENTS.exists():
        return records
    for path in sorted(SEGMENTS.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rec["_file"] = path.name
                records.append(rec)
    return records


def print_stats(title: str, stats: dict, path: Path | None = None) -> None:
    print(f"\n{title}")
    if path:
        print(f"  wrote {path}  ({path.stat().st_size/1024:.0f} KB)")
    ordered = sorted(stats.items(), key=lambda kv: (not kv[0].startswith("kept"), kv[0]))
    for key, value in ordered:
        if value and not key.startswith("kept_words_"):
            print(f"  {key:<24} {value:,}")
    for key, value in ordered:
        if key.startswith("kept_words_") and value:
            print(f"  {key.replace('kept_words_', 'words / '):<24} {value:,}")
