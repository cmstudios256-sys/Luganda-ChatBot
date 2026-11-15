#!/usr/bin/env python3
"""Ingest the Luganda Bible — the biggest chunk of attested, published Luganda there is.

    # one folder of per-book text files, verse numbers at the start of each verse
    python -m corpus.ingest_bible bayibuli/ --out bible

    # same, plus English beside it: also writes verse-aligned translation pairs
    python -m corpus.ingest_bible bayibuli/ --parallel english/ --out bible

    # what did it actually get?
    python -m corpus.ingest_bible bayibuli/ --dry-run

Accepted verse layout (line starts with chapter:verse, book name optional):

    1:1 Olw'okusooka Katonda yatonda eggulu n'ensi.
    Olw'okusooka 1:2 Ensi yali tekyali kintu…
    3:16 Kubanga Katonda bwe yayagala ensi nnyo…

Two things this script deliberately does *not* do:

1. **It does not pretend the Bible is the same register as a letter or a chat.**
   Every record is tagged ``register: biblical`` so the trainer can cap its share of the
   mix. A model that learns "Awo ne kiwedde" as its only reply style sounds like a
   pulpit, not a friend on WhatsApp.
2. **It does not silently trust a file it cannot parse.** If no verse markers are found
   the script says so and stops, rather than dumping a blob into the corpus.

Why records from here come in as ``verified: true``: this is a published translation
made by named translators and reviewed by a Bible society. That is the definition of
attested Luganda. The *old* (1968) edition is public domain; the Biblica
"Bayibuli Entukuvu" is CC BY-SA 4.0 (attribution + share-alike), and the 2003 Bible
Society of Uganda edition is all rights reserved — check which file you have and pass
``--license`` so the record keeps its own provenance. See CORPUS-PLAN.md.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from .pipeline import OUT, print_stats, process, write_segments
from .text_extract import Segment

# "3:16 text", "1 Oku 3:16 text", "Olw'okusooka 3:16 text", "3.16 text"
VERSE_RE = re.compile(
    r"^\s*(?:(?P<book>[1-3]?\s?[^\W\d_][\w'’\-]*(?:\s+[^\W\d_][\w'’\-]*)?)\s+)?"
    r"(?P<chapter>\d{1,3})\s*[:.]\s*(?P<verse>\d{1,3})\s+(?P<text>\S.*)$",
    re.UNICODE,
)
# running heads / page numbers that survive copy-paste
NOISE_RE = re.compile(r"^\s*(?:page\s+\d+|\d{1,3}|baibuli|bayibuli)\s*$", re.IGNORECASE)


def book_name(path: Path) -> str:
    stem = path.stem.replace("_", " ").replace("-", " ").strip()
    return re.sub(r"^\d{1,2}[\s.]+", "", stem) or path.stem   # "01 Olwokusooka" -> "Olwokusooka"


def parse_verses(path: Path) -> tuple[list[dict], int, int]:
    """Return [{book, chapter, verse, text}], (verses found, lines skipped)."""
    verses: list[dict] = []
    skipped = 0
    book = book_name(path)
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw.strip() or NOISE_RE.match(raw):
            continue
        m = VERSE_RE.match(raw)
        if not m:
            # a wrapped continuation of the previous verse is common in PDF text
            if verses and not raw.strip().startswith(("Chapter", "Essuula", "©", "Copyright")):
                verses[-1]["text"] += " " + raw.strip()
            else:
                skipped += 1
            continue
        if m.group("book"):
            book = m.group("book").strip()
        verses.append({
            "book": book,
            "chapter": int(m.group("chapter")),
            "verse": int(m.group("verse")),
            "text": m.group("text").strip(),
        })
    return verses, len(verses), skipped


def group_verses(verses: list[dict], *, words_target: int = 120) -> list[Segment]:
    """Pack consecutive verses into readable paragraphs.

    Chunking mid-chapter rather than per-verse matters: a single verse is 20-25 words,
    which is too short to teach style and gets rejected by the word filter anyway.
    """
    segments: list[Segment] = []
    buf: list[dict] = []
    words = 0

    def flush() -> None:
        nonlocal buf, words
        if not buf:
            return
        first, last = buf[0], buf[-1]
        ref = f"{first['book']} {first['chapter']}:{first['verse']}"
        if len(buf) > 1:
            ref += f"-{last['verse']}" if last["chapter"] == first["chapter"] else f"-{last['chapter']}:{last['verse']}"
        segments.append(Segment(
            text=" ".join(v["text"] for v in buf),
            source=f"bible:{first['book']}",
            kind="bible",
            locator=ref,
            meta={"register": "biblical", "verses": len(buf)},
        ))
        buf, words = [], 0

    prev = None
    for v in verses:
        if prev is not None:
            same_chapter_run = (v["book"], v["chapter"]) == (prev["book"], prev["chapter"]) \
                and v["verse"] == prev["verse"] + 1
            if not same_chapter_run:
                flush()
        buf.append(v)
        words += len(v["text"].split())
        prev = v
        if words >= words_target:
            flush()
    flush()
    return segments


def collect(folder: Path) -> list[Path]:
    if folder.is_file():
        return [folder]
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in {".txt", ".md", ".csv"})


def write_parallel(lug_files: list[Path], eng_folder: Path, out_path: Path) -> dict:
    """Verse-aligned EN->LG pairs, aligned by book position + chapter:verse.

    Book files sort the same way in both folders when they are numbered 01..66, which is
    how every Bible download ships them. If the two sides disagree we report it instead
    of guessing.
    """
    eng_files = collect(eng_folder)
    if len(eng_files) != len(lug_files):
        print(f"! {len(lug_files)} Luganda book files vs {len(eng_files)} English — "
              "aligning by sorted position, unmatched books are skipped.")
    pairs: list[dict] = []
    for lug_path, eng_path in zip(lug_files, eng_files):
        lug, _, _ = parse_verses(lug_path)
        eng, _, _ = parse_verses(eng_path)
        eng_by_key = {(v["chapter"], v["verse"]): v["text"] for v in eng}
        for v in lug:
            english = eng_by_key.get((v["chapter"], v["verse"]))
            if not english:
                continue
            pairs.append({
                "luganda": v["text"],
                "english": english,
                "ref": f"{v['book']} {v['chapter']}:{v['verse']}",
                "verified": True,
                "source": "bible-parallel",
            })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
    return {"pairs": len(pairs), "path": str(out_path)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="folder of per-book .txt files (or a single file)")
    ap.add_argument("--out", default="bible", help="name of the segment file to write")
    ap.add_argument("--parallel", metavar="ENGLISH_FOLDER",
                    help="also write verse-aligned EN->LG pairs from a matching English folder")
    ap.add_argument("--pairs-out", default=str(OUT / "pairs" / "bible_parallel.jsonl"))
    ap.add_argument("--license", default="unknown — check the edition you downloaded",
                    help="e.g. 'public domain (UBS 1968)' or 'CC BY-SA 4.0 Biblica 2014'")
    ap.add_argument("--words-target", type=int, default=120, help="rough words per paragraph")
    ap.add_argument("--min-words", type=int, default=25)
    ap.add_argument("--no-verify", action="store_true",
                    help="mark records unverified (use if you are unsure the text is a real published edition)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    folder = Path(args.folder).expanduser()
    if not folder.exists():
        print(f"No such path: {folder}")
        return 1
    files = collect(folder)
    if not files:
        print("No .txt/.md/.csv files in there.")
        return 1

    segments: list[Segment] = []
    total_verses = total_skipped = 0
    for path in files:
        verses, found, skipped = parse_verses(path)
        total_verses += found
        total_skipped += skipped
        if not found:
            print(f"! {path.name}: no verse markers found — skipped (is this the right format?)")
            continue
        got = group_verses(verses, words_target=args.words_target)
        for seg in got:
            seg.meta.update({
                "verified": not args.no_verify,
                "license": args.license,
                "edition": path.stem,
            })
        segments.extend(got)
        print(f"  {path.name}: {found:,} verses -> {len(got):,} paragraphs "
              f"({sum(len(s.text.split()) for s in got):,} words)")

    print(f"\n{total_verses:,} verses parsed, {total_skipped:,} unparsed line(s) skipped.")
    if args.dry_run:
        return 0
    if not segments:
        print("Nothing parsed. Expected lines starting like '1:1 Olw'okusooka …'.")
        return 1

    kept, stats = process(segments, min_words=args.min_words)
    path = write_segments(args.out, kept)
    print_stats("Bible ingest", stats, path)
    if stats.get("rejected_not_luganda") or stats.get("rejected_english"):
        print(f"! {stats.get('rejected_not_luganda', 0) + stats.get('rejected_english', 0)} paragraph(s) "
              "were not recognised as Luganda — usually means English crept into the file.")

    if args.parallel:
        pairs = write_parallel(files, Path(args.parallel).expanduser(), Path(args.pairs_out))
        print(f"\nParallel pairs: {pairs['pairs']:,} -> {pairs['path']}")
        print("  Use these for translation SFT, not for chat style.")

    print("\nReminder: this is biblical register. Keep it a minority of any fine-tuning "
          "mix (a quarter at most) or the bot starts answering like a pulpit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
