#!/usr/bin/env python3
"""Ingest textbooks, handbooks, newspapers and subtitle files from disk.

    python -m corpus.ingest_files books/ handbooks/ newspapers/ --out books
    python -m corpus.ingest_files ~/scans --dry-run          # see what it would read
    python -m corpus.ingest_files tv/ --include-english      # keep the code-mixed stuff

Folder layout it expects (any of these names, or pass your own paths):
    books/        textbooks, novels — clean modern written Luganda
    handbooks/    guides, manuals, government/NGO booklets
    newspapers/   Bukedde / New Vision print scans or PDFs
    tv/           .srt / .vtt subtitle files exported from shows

Scanned PDFs with no text layer are reported, not silently dropped: they need OCR
first. See CORPUS-PLAN.md for the OCR recipe.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pipeline import print_stats, process, write_segments
from .text_extract import SUPPORTED, extract, split_big_segment


def collect(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED))
        elif path.is_file():
            files.append(path)
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="files or folders")
    ap.add_argument("--out", default="files", help="name of the segment file to write")
    ap.add_argument("--min-words", type=int, default=25)
    ap.add_argument("--min-luganda-score", type=float, default=0.05)
    ap.add_argument("--keep-mixed", action="store_true", default=True,
                    help="keep code-mixed Luganda/English (default on, flagged)")
    ap.add_argument("--drop-mixed", dest="keep_mixed", action="store_false")
    ap.add_argument("--include-english", action="store_true", help="keep English text too (rarely wanted)")
    ap.add_argument("--verified", action="store_true",
                    help="you, a native speaker, attest this text is correct Luganda — sets verified:true")
    ap.add_argument("--dry-run", action="store_true", help="list what would be read, extract nothing")
    args = ap.parse_args()

    files = collect([Path(p) for p in args.paths])
    if not files:
        print("No supported files found.")
        return 1
    print(f"{len(files)} file(s):")
    for f in files[:40]:
        print(f"  {f}  ({f.stat().st_size/1024:.0f} KB)")
    if len(files) > 40:
        print(f"  … and {len(files) - 40} more")
    if args.dry_run:
        return 0

    segments, failures = [], []
    for path in files:
        try:
            got = extract(path)
        except Exception as exc:                      # noqa: BLE001
            failures.append(f"{path}: {exc}")
            continue
        for seg in got:
            if args.verified:
                seg.meta["verified"] = True
            segments.extend(split_big_segment(seg))
        print(f"  {path.name}: {sum(len(s.text.split()) for s in got):,} words in {len(got)} segment(s)")

    if failures:
        print("\nCould not read:")
        for f in failures:
            print("  -", f)

    kept, stats = process(
        segments,
        min_words=args.min_words,
        keep_english=args.include_english,
        keep_mixed=args.keep_mixed,
        min_luganda_score=args.min_luganda_score,
    )
    path = write_segments(args.out, kept)
    print_stats("Extraction result", stats, path)

    needs_ocr = [s for s in segments if s.meta.get("needs_ocr")]
    if needs_ocr:
        print(f"\n! {len(needs_ocr)} page(s) had no text layer — they are scans.")
        print("  OCR them first (see CORPUS-PLAN.md), then re-run this script.")
    if not kept:
        print("\nNothing usable came out. Either the language filter is rejecting the files"
              " (check with --min-luganda-score 0) or the text really is not Luganda.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
