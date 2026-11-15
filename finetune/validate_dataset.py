#!/usr/bin/env python3
"""Validate a Luganda chat/SFT dataset before you spend money on GPUs.

Checks structural things (schema, role order) and language things a generic
validator will not catch for Luganda: stray characters, English leakage, broken
gemination, apostrophe style, and near-duplicates across splits.

Usage:
    python finetune/validate_dataset.py finetune/data/seed_luganda.jsonl
    python finetune/validate_dataset.py finetune/out --strict
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

VALID_ROLES = {"system", "user", "assistant"}
# Characters that should never appear in Luganda text.
FORBIDDEN_CHARS = set("ŋŊ")
# Words that suggest the "Luganda" answer is actually English.
ENGLISH_MARKERS = {
    "the", "and", "is", "are", "was", "were", "you", "your", "have", "has", "had",
    "this", "that", "with", "for", "from", "not", "will", "would", "should", "can",
    "could", "about", "because", "there", "their", "what", "when", "where", "how",
    "please", "sorry", "hello", "thanks", "thank", "very", "much", "good", "bad",
    "want", "need", "know", "think", "make", "going", "get", "just", "also",
}
# Common misspellings that betray a machine writer. Extend with your own review.
SPELLING_TRAPS = {
    r"\bamaji\b": "amazzi (water)",
    r"\bssomero\b": None,          # correct
    r"\bsomero\b": "ssomero (school) — geminate ss",
    r"\bnyumba\b": "ennyumba (house) — geminate nn",
    r"\bente\b": None,
    r"\babaana\b": None,
    r"\bweebale\b": None,
    r"\bwebale\b": "weebale (thank you) — doubled ee",
    r"\boluganda\b": "Oluganda (capital O as a language name)",
    r"\bkyeyagalire\b": None,
    r"\bntegerede\b": "ntegeeredde / ntegedde — check the perfect form",
    r"\b[a-z]\s+ŋ": "no eng character in standard Luganda orthography",
}
SPELLING_TRAPS = {k: v for k, v in SPELLING_TRAPS.items() if v}


def norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    return re.sub(r"[^a-z' ]+", " ", text).strip()


def load_records(path: Path) -> list[dict]:
    records = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            records.append({"_error": f"line {i}: invalid JSON ({exc.msg})", "_line": i})
            continue
        obj["_line"] = i
        records.append(obj)
    return records


def check_record(rec: dict) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if "_error" in rec:
        return [rec["_error"]], []

    # DPO schema (prompt/chosen/rejected) is a different file from SFT and validates on its
    # own terms. Counting it as 5 "missing messages" errors made the whole suite look broken.
    if "prompt" in rec and ("chosen" in rec or "rejected" in rec):
        errors, warnings = [], []
        if not str(rec.get("prompt", "")).strip():
            errors.append("empty 'prompt'")
        for key in ("chosen", "rejected"):
            if not str(rec.get(key, "")).strip():
                errors.append(f"empty '{key}'")
            elif FORBIDDEN_CHARS & set(str(rec[key])):
                errors.append(f"forbidden character in {key}")
        if str(rec.get("chosen", "")).strip() == str(rec.get("rejected", "")).strip():
            errors.append("chosen and rejected are identical — nothing to learn")
        return errors, warnings

    # Eval-set schema (prompt/reference) is valid too — it never trains, it scores.
    if "prompt" in rec and "reference" in rec:
        if not str(rec["prompt"]).strip():
            errors.append("empty 'prompt'")
        if not str(rec["reference"]).strip():
            errors.append("empty 'reference'")
        if FORBIDDEN_CHARS & set(str(rec["reference"])):
            errors.append("forbidden character in reference")
        if rec.get("verified") is False:
            warnings.append("eval item not verified by a native speaker")
        return errors, warnings

    msgs = rec.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return ["missing or empty 'messages'"], []
    if msgs[0].get("role") not in {"system", "user"}:
        errors.append("first message must be 'system' or 'user'")
    if msgs[-1].get("role") != "assistant":
        errors.append("last message must be 'assistant' (nothing to learn otherwise)")

    expect = "user"
    for j, m in enumerate(msgs):
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role not in VALID_ROLES:
            errors.append(f"msg {j}: bad role {role!r}")
            continue
        if not content:
            errors.append(f"msg {j}: empty content")
            continue
        if role == "system":
            continue
        if role != expect and role != "system":
            warnings.append(f"msg {j}: expected {expect}, found {role}")
        expect = "assistant" if role == "user" else "user"
        if len(content) > 2000:
            warnings.append(f"msg {j}: {len(content)} chars is long for chat")
        if FORBIDDEN_CHARS & set(content):
            errors.append(f"msg {j}: forbidden character(s) {sorted(FORBIDDEN_CHARS & set(content))}")
        for pattern, hint in SPELLING_TRAPS.items():
            if re.search(pattern, content.lower()):
                warnings.append(f"msg {j}: {hint}")
        # English leakage, but only for the Luganda side of the pair
        words = re.findall(r"[a-zA-Z']+", content.lower())
        if len(words) >= 6:
            eng = sum(1 for w in words if w in ENGLISH_MARKERS)
            ratio = eng / len(words)
            if ratio > 0.25:
                warnings.append(f"msg {j}: {ratio:.0%} English function words — is this really Luganda?")
        if content.count("?") > 3:
            warnings.append(f"msg {j}: more than 3 question marks")
    if rec.get("verified") is False:
        warnings.append("not verified by a native speaker")
    return errors, warnings


def detect_duplicates(records: list[dict]) -> tuple[list[tuple[str, int, int]], list[tuple[str, int, int]]]:
    exact: dict[str, list[int]] = {}
    for rec in records:
        if "_error" in rec:
            continue
        key = norm(" ".join(m.get("content", "") for m in rec.get("messages", [])))
        exact.setdefault(key, []).append(rec["_line"])
    dups = [(k, v[0], v[1]) for k, v in exact.items() if len(v) > 1]
    near = []
    seen: dict[str, int] = {}
    for rec in records:
        if "_error" in rec:
            continue
        users = [m.get("content", "") for m in rec.get("messages", []) if m.get("role") == "user"]
        if not users:
            continue
        key = " ".join(norm(users[0]).split()[:8])
        if key in seen:
            near.append((key, seen[key], rec["_line"]))
        else:
            seen[key] = rec["_line"]
    return dups, near


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help=".jsonl files or directories")
    ap.add_argument("--strict", action="store_true", help="treat warnings as failures")
    ap.add_argument("--max-unverified", type=float, default=1.0,
                    help="fail if this fraction of records is unverified (e.g. 0.2)")
    args = ap.parse_args()

    files: list[Path] = []
    for raw in args.paths:
        p = Path(raw)
        if p.is_dir():
            files.extend(sorted(p.rglob("*.jsonl")))
        elif p.exists():
            files.append(p)
        else:
            print(f"!! not found: {p}")

    total_err = total_warn = total_rec = 0
    for path in files:
        records = load_records(path)
        errs, warns = 0, 0
        for rec in records:
            e, w = check_record(rec)
            errs += len(e)
            warns += len(w)
            if e or (args.strict and w):
                print(f"\n[{path.name}:{rec.get('_line')}]")
                for line in e:
                    print("  ERROR  ", line)
                for line in w[:6]:
                    print("  warn   ", line)
        dups, near = detect_duplicates(records)
        unverified = sum(1 for r in records if r.get("verified") is False)
        roles = Counter()
        for rec in records:
            for m in rec.get("messages", []):
                roles[m.get("role")] += 1
        words = sum(len((m.get("content") or "").split()) for r in records for m in r.get("messages", []))
        frac = unverified / max(len(records), 1)
        print(
            f"\n{path}\n  records      {len(records)}\n  words        {words}"
            f"\n  errors       {errs}\n  warnings     {warns}"
            f"\n  exact dups   {len(dups)}\n  near dups    {len(near)}"
            f"\n  unverified   {unverified}/{len(records)} ({frac:.0%})"
            f"\n  roles        {dict(roles)}"
        )
        for key, a, b in dups[:5]:
            print(f"  dup lines {a} & {b}: {key[:70]}")
        for key, a, b in near[:5]:
            print(f"  near lines {a} & {b}: {key[:70]}")
        if frac > args.max_unverified:
            print(f"  FAIL: {frac:.0%} unverified > --max-unverified {args.max_unverified:.0%}")
            errs += 1
        total_err += errs
        total_warn += warns
        total_rec += len(records)

    print(f"\n=== {total_rec} records, {total_err} errors, {total_warn} warnings across {len(files)} file(s)")
    if total_err:
        print("Fix the errors, then run build_dataset.py. Warnings are the queue for a native reviewer.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
