#!/usr/bin/env python3
"""Merge every ingested segment into one corpus, then emit the two things it is for.

    python -m corpus.build_corpus                          # report + all outputs
    python -m corpus.build_corpus --push-to-app http://localhost:8000
    python -m corpus.build_corpus --retrieval-only         # skip the CPT files

Outputs in corpus/out/:
    corpus.jsonl            every kept segment, deduped, with provenance
    retrieval/docs.jsonl    documents shaped for POST /api/knowledge (grounding, instant)
    cpt/train.txt cpt/val.txt   plain text for continued pre-training (slow, expensive)
    report.json             counts by kind/modality — the honest scoreboard

Which one you actually need:
  * Newspapers, prices, current affairs  → retrieval. Facts change weekly; you cannot
    fine-tune daily, and a model trained on yesterday's prices confidently lies.
  * Textbooks, handbooks                 → retrieval now, continued pre-training later.
  * TV / TikTok transcripts              → register and style for SFT, never for facts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import httpx

from .pipeline import OUT, content_hash, normalize, read_all_segments

SEPARATOR = "\n\n<<<DOC>>>\n\n"


def near_dup_key(text: str, shingle: int = 5) -> str:
    """Cheap boilerplate killer: hash of the first shingle of words + a length bucket.
    Catches syndicated copies of the same article far better than exact hashing."""
    words = re.findall(r"[a-z']+", text.lower())[:shingle]
    return hashlib.sha1((" ".join(words) + f"|{len(text)//200}").encode()).hexdigest()[:12]


def strip_repeated_paragraphs(records: list[dict], *, header_max_words: int = 25,
                              threshold: int = 3) -> tuple[list[dict], Counter]:
    """Remove boilerplate without eating content.

    Two different problems, two different rules:
      * Very short units repeating 3+ times ("© Bukedde", page number, running head)
        are boilerplate — drop every copy.
      * Longer units that repeat are duplicate content (a syndicated column, a page
        reprinted twice) — keep the first copy, drop the rest.
    """
    stats: Counter = Counter()
    units_per_record = [re.split(r"\n+", rec["text"]) for rec in records]

    counts: Counter = Counter()
    for units in units_per_record:
        for unit in units:
            if len(unit.split()) >= 3:
                counts[content_hash(unit)] += 1

    boilerplate = {
        h for h, n in counts.items()
        if n >= threshold and _words_of(h, units_per_record) <= header_max_words
    }
    seen: set[str] = set()
    out: list[dict] = []
    for rec, units in zip(records, units_per_record):
        kept: list[str] = []
        for unit in units:
            if len(unit.split()) < 3:
                continue
            digest = content_hash(unit)
            if digest in boilerplate:
                stats["boilerplate_units_removed"] += 1
                continue
            if digest in seen:
                stats["duplicate_units_removed"] += 1
                continue
            seen.add(digest)
            kept.append(unit)
        text = "\n".join(kept).strip()
        words = len(text.split())
        original_words = len(rec["text"].split())
        if words < 10 or (original_words and words / original_words < 0.4):
            # The heuristic ate too much (a page that genuinely repeats, a table).
            # Keep the original and flag it for a human instead of deleting content.
            stats["stripping_skipped_suspicious"] += 1
            out.append(rec)
            continue
        out.append({**rec, "text": text, "words": words} if words != rec.get("words") else rec)
    return out, stats


def _words_of(digest: str, units_per_record: list[list[str]]) -> int:
    """Word count of the first unit with this hash (cheap, only used for classification)."""
    for units in units_per_record:
        for unit in units:
            if content_hash(unit) == digest:
                return len(unit.split())
    return 10**6


def dedupe(records: list[dict]) -> tuple[list[dict], Counter]:
    kept: list[dict] = []
    stats: Counter = Counter()
    seen_exact: set[str] = set()
    seen_near: set[str] = set()
    # Prefer longer, text-first segments when duplicates collide.
    records.sort(key=lambda r: (-r.get("words", 0), r.get("kind", "")))
    for rec in records:
        exact = rec.get("id") or content_hash(rec["text"])
        if exact in seen_exact:
            stats["dropped_exact_duplicate"] += 1
            continue
        near = near_dup_key(rec["text"])
        if near in seen_near:
            stats["dropped_near_duplicate"] += 1
            continue
        seen_exact.add(exact)
        seen_near.add(near)
        kept.append(rec)
    stats["kept"] = len(kept)
    return kept, stats


def retrieval_docs(records: list[dict], *, max_chars: int = 6000) -> list[dict]:
    """One document per source page/article, so retrieval returns whole contexts."""
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for rec in records:
        grouped[(rec["source"], rec.get("locator", ""))].append(rec)
    docs = []
    for (source, locator), recs in grouped.items():
        text = "\n\n".join(r["text"] for r in recs)[:max_chars]
        kind = recs[0].get("kind", "unknown")
        title = Path(source.split(":", 1)[-1]).name if ":" in source else source
        docs.append({
            "title": f"{kind}: {title} {locator}".strip()[:180],
            "text": text,
            "source": source,
            "modality": recs[0].get("modality", "text"),
            "verified": all(r.get("verified") for r in recs),
        })
    return docs


def cpt_chunks(records: list[dict], *, words_per_chunk: int = 1200) -> list[str]:
    """Plain-text stream for continued pre-training. Text-only: never mix chat turns
    or retrieval markers into this, or the model learns to emit them."""
    chunks: list[str] = []
    for rec in records:
        if rec.get("modality") == "spoken" and not rec.get("verified"):
            continue                     # unverified ASR does not belong in CPT text
        text = normalize(rec["text"])
        words = text.split()
        for i in range(0, len(words), words_per_chunk):
            chunk = " ".join(words[i:i + words_per_chunk]).strip()
            if len(chunk.split()) >= 120:
                chunks.append(chunk)
    return chunks


def write_cpt(chunks: list[str], out_dir: Path, *, val_fraction: float = 0.05) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(chunks) <= 1:
        val, train = [], chunks            # never emit an empty training file
    else:
        split = max(1, int(len(chunks) * val_fraction))
        val, train = chunks[:split], chunks[split:]
        if not train:                       # tiny corpus: keep at least one chunk to train on
            train, val = val, []
    (out_dir / "train.txt").write_text((SEPARATOR).join(train) + "\n", encoding="utf-8")
    (out_dir / "val.txt").write_text((SEPARATOR).join(val) + "\n", encoding="utf-8")
    return {"train_chunks": len(train), "val_chunks": len(val),
            "train_words": sum(len(c.split()) for c in train),
            "val_words": sum(len(c.split()) for c in val)}


def push_to_app(url: str, docs: list[dict], *, batch: int = 200) -> dict:
    url = url.rstrip("/") + "/api/knowledge/bulk"
    added, chunks = 0, 0
    with httpx.Client(timeout=120) as client:
        for i in range(0, len(docs), batch):
            payload = {"documents": [
                {"title": d["title"], "text": d["text"], "source": d["source"]}
                for d in docs[i:i + batch]
            ]}
            resp = client.post(url, json=payload)
            if resp.status_code >= 400:
                raise SystemExit(f"push failed at batch {i//batch}: HTTP {resp.status_code} {resp.text[:200]}")
            data = resp.json()
            added += data.get("added", 0)
            chunks += data.get("chunks_added", 0)
            print(f"  pushed {added}/{len(docs)} documents…")
    return {"documents": added, "chunks": chunks}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--retrieval-only", action="store_true", help="skip the CPT text files")
    ap.add_argument("--push-to-app", default="", help="base URL, e.g. http://localhost:8000")
    ap.add_argument("--max-retrieval-docs", type=int, default=20000)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = read_all_segments()
    if not records:
        print("No segments found. Run ingest_files / ingest_web / ingest_media first.")
        return 1
    print(f"{len(records):,} segments in from {len({r.get('_file') for r in records})} ingest file(s)")

    records, boiler_stats = strip_repeated_paragraphs(records)
    if boiler_stats:
        print(f"boilerplate: {dict(boiler_stats)}")
    kept, dedupe_stats = dedupe(records)
    with (out_dir / "corpus.jsonl").open("w", encoding="utf-8") as fh:
        for rec in kept:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    by_kind: Counter = Counter()
    by_modality: Counter = Counter()
    words_kind: Counter = Counter()
    for rec in kept:
        by_kind[rec.get("kind", "unknown")] += 1
        by_modality[rec.get("modality", "text")] += 1
        words_kind[rec.get("kind", "unknown")] += rec.get("words", 0)

    total_words = sum(r.get("words", 0) for r in kept)
    code_mixed = sum(1 for r in kept if r.get("lang", {}).get("code_mix"))
    spoken = by_modality.get("spoken", 0)
    tokens_est = int(total_words * 1.6)

    docs = retrieval_docs(kept)[: args.max_retrieval_docs]
    ret_dir = out_dir / "retrieval"
    ret_dir.mkdir(parents=True, exist_ok=True)
    with (ret_dir / "docs.jsonl").open("w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")

    cpt_info = {}
    if not args.retrieval_only:
        chunks = cpt_chunks(kept)
        cpt_info = write_cpt(chunks, out_dir / "cpt") if chunks else {"note": "no text-only segments for CPT"}

    report = {
        "segments_in": len(records),
        "segments_kept": len(kept),
        "dedupe": dict(dedupe_stats),
        "boilerplate": dict(boiler_stats),
        "words_total": total_words,
        "tokens_estimate": tokens_est,
        "by_kind": dict(by_kind),
        "words_by_kind": dict(words_kind),
        "by_modality": dict(by_modality),
        "code_mixed_segments": code_mixed,
        "spoken_segments": spoken,
        "retrieval_documents": len(docs),
        "cpt": cpt_info,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== corpus report ===")
    print(f"  segments kept        {len(kept):,} of {len(records):,}")
    print(f"  words                {total_words:,}")
    print(f"  tokens (est.)        {tokens_est:,}")
    print(f"  by kind              {dict(by_kind)}")
    print(f"  by modality          {dict(by_modality)}")
    print(f"  code-mixed segments  {code_mixed:,}  ({code_mixed/max(len(kept),1):.0%})")
    print(f"  dedupe               {dict(dedupe_stats)}")
    print(f"  retrieval documents  {len(docs):,}")
    if cpt_info:
        print(f"  CPT text             {cpt_info}")

    # Honest interpretation, in the report itself so nobody has to remember it.
    print("\n=== what this corpus can actually do ===")
    if total_words < 50_000:
        print("  <50k words: enough to ground a bot with retrieval. Not enough to change a model.")
    elif total_words < 1_000_000:
        print("  50k–1M words: strong for retrieval and for SFT style. Continued pre-training would")
        print("  be a rounding error — a 14B model sees billions of tokens; this is thousands.")
    else:
        print("  >1M words: real ground for domain-adaptive pre-training. Budget GPUs and weeks.")
    if spoken:
        print(f"  {spoken:,} spoken segments: use them to teach register and vocabulary, and keep")
        print("  them out of any spelling or grammar target until a native speaker listens.")

    if args.push_to_app:
        print(f"\nPushing {len(docs):,} documents into the app's knowledge base at {args.push_to_app}")
        result = push_to_app(args.push_to_app, docs)
        print(f"  done: {result}")
        print("  Chat now retrieves these facts — no training run involved.")

    print(f"\nwrote {out_dir}/corpus.jsonl, {ret_dir}/docs.jsonl, report.json"
          + ("" if args.retrieval_only else ", cpt/train.txt, cpt/val.txt"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
