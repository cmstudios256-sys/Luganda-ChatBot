#!/usr/bin/env python3
"""Fix the things that are cheap to fix, drop what cannot be fixed, report both.

    python3 -m finetune.clean_dataset finetune/data --dry-run
    python3 -m finetune.clean_dataset finetune/data

Real example from this project: the public `alpaca_4k_luganda` set contains `ŋ` (eng), a
Lusoga/Runyankole letter that Luganda does not use. Left in, it teaches the model a character
that is wrong in every Luganda sentence it appears in — and a validator already flags it, so
the fix is mechanical: `ŋ` → `ng`, `ɲ` → `ny`, curly quotes → straight, zero-width junk out,
whitespace collapsed.

What it will **not** do: rewrite the Luganda. Style, agreement, register and meaning need a
native speaker (`corpus/make_review_sheet.py`), and a script that "improves" Luganda is how
you get a corpus that is confidently wrong everywhere.

Every file is backed up to `*.bak` before being rewritten, and the counts are printed so you
can see exactly how much of your dataset was actually touched.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

TRANSLITERATE = {
    "\u014b": "ng",      # ŋ  eng        -> ng
    "\u014b\u0301": "ng",
    "\u0272": "ny",      # ɲ  ny         -> ny
    "\u2018": "'", "\u2019": "'", "\u02bc": "'",
    "\u201c": '"', "\u201d": '"',
    "\u200b": "", "\u200c": "", "\u200d": "", "\ufeff": "",
    "\u00a0": " ",
}
FORBIDDEN_LEFT = set("\u014b\u0272\u200b\u200c\u200d\ufeff")


def clean_text(text: str) -> str:
    for bad, good in TRANSLITERATE.items():
        text = text.replace(bad, good)
    text = " ".join(text.split())
    return text.strip()


def clean_record(rec: dict) -> tuple[dict | None, bool]:
    """Return (record or None if unusable, changed?).

    Records without ``messages`` are passed through untouched — a cleaner that silently eats
    the eval set is worse than the problem it was written to solve.
    """
    msgs = rec.get("messages")
    if not isinstance(msgs, list) or not msgs:
        # eval sets (prompt/reference) and DPO pairs (prompt/chosen/rejected) have no chat
        # turns; they are not this tool's business and must come out untouched.
        return rec, False
    changed = False
    out = []
    for msg in msgs:
        content = msg.get("content") or ""
        cleaned = clean_text(content)
        if cleaned != content:
            changed = True
        if not cleaned:
            return None, changed        # an empty turn makes the whole example unusable
        out.append({**msg, "content": cleaned})
    if out[-1].get("role") != "assistant":
        return None, changed            # nothing to learn from
    if any(FORBIDDEN_LEFT & set(m["content"]) for m in out):
        return None, changed
    return {**rec, "messages": out}, changed


def process_file(path: Path, *, write: bool) -> dict:
    lines = path.read_text(encoding="utf-8").splitlines()
    kept, changed = [], 0
    dropped = Counter()
    for line in lines:
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            dropped["invalid json"] += 1
            continue
        fixed, was_changed = clean_record(rec)
        if fixed is None:
            dropped["unusable after cleaning"] += 1
            continue
        kept.append(fixed)
        changed += bool(was_changed)
    if write and (changed or dropped):
        backup = path.with_suffix(path.suffix + ".bak")
        if not backup.exists():
            backup.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with path.open("w", encoding="utf-8") as fh:
            for rec in kept:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"records_in": len(lines), "records_out": len(kept), "changed": changed,
            "dropped": dict(dropped)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help=".jsonl files or folders")
    ap.add_argument("--dry-run", action="store_true", help="report only, change nothing")
    args = ap.parse_args()

    files: list[Path] = []
    for raw in args.paths:
        path = Path(raw)
        files.extend(sorted(path.rglob("*.jsonl")) if path.is_dir() else [path])

    total_changed = total_dropped = 0
    for path in files:
        stats = process_file(path, write=not args.dry_run)
        total_changed += stats["changed"]
        total_dropped += sum(stats["dropped"].values())
        flag = "would change" if args.dry_run else "changed"
        print(f"{path}\n  {stats['records_in']} in, {stats['records_out']} out, "
              f"{flag} {stats['changed']}"
              + (f", dropped {sum(stats['dropped'].values())} ({dict(stats['dropped'])})"
                 if stats["dropped"] else ""))

    print(f"\n{total_changed} record(s) cleanable, {total_dropped} unusable"
          + (" — nothing written (dry run)" if args.dry_run else ""))
    if not args.dry_run and (total_changed or total_dropped):
        print("Backups are next to each file as *.jsonl.bak.")
        print("Now: python3 finetune/validate_dataset.py finetune/data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
