#!/usr/bin/env python3
"""Turn verse-aligned Bible pairs into SFT records — carefully, and in a minority.

    python3 -m finetune.ingest_bible_pairs --dry-run
    python3 -m finetune.ingest_bible_pairs                    # -> finetune/data/ingested/
    python3 -m finetune.ingest_bible_pairs --both-directions --max-share 0.15

Input is what `corpus/ingest_bible.py --parallel` writes: one JSON object per verse,
`{"luganda": …, "english": …, "ref": "Olw'okusooka 1:1"}`.

Two decisions are baked in, and both are about not ruining a good chat model:

1. **A translation record is not a conversation record.** Teaching "Vvuunula kino mu
   Luganda:" 30,000 times produces a translator that answers "oli otya?" with a translation.
   So the output is capped at a *share* of the existing SFT set (default 20%), and the cap is
   computed from what is already on disk, not guessed.
2. **Instruction phrasing varies.** Three templates, picked deterministically from the verse
   reference, so the model does not learn one magic sentence and fail on a paraphrase.

Verses are short and formal: this is good for vocabulary, morphology and the meaning of
`ss`/`nn`, and it is not a substitute for the letters and conversation that teach *voice*.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DEFAULT_PAIRS = ROOT.parent / "corpus" / "out" / "pairs" / "bible_parallel.jsonl"
DEFAULT_OUT = DATA / "ingested" / "bible_translation.jsonl"

TO_LUGANDA = [
    "Vvuunula kino mu Luganda: {source}",
    "Kino kivvuunule mu Luganda: {source}",
    "Nnyambeko okuvvuunula kino mu Luganda: {source}",
]
TO_ENGLISH = [
    "Vvuunula kino mu Lungereza: {source}",
    "Kino kivvuunule mu Lungereza: {source}",
]


def load_pairs(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"No pairs at {path}. Build them first:\n"
                         "  python3 -m corpus.ingest_bible bayibuli/ --parallel english/")
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        pair = json.loads(line)
        if pair.get("luganda") and pair.get("english"):
            pairs.append(pair)
    return pairs


def existing_sft_records(exclude: Path) -> int:
    """How many chat records already exist in finetune/data (the thing we must not swamp)."""
    total = 0
    for path in sorted(DATA.rglob("*.jsonl")):
        if path == exclude or path.name.startswith("bible_translation"):
            continue
        if "eval" in path.parts:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    if "messages" in json.loads(line):
                        total += 1
                except json.JSONDecodeError:
                    continue
    return total


def pick_template(ref: str, templates: list[str]) -> str:
    digest = hashlib.sha256(ref.encode("utf-8")).hexdigest()
    return templates[int(digest[:8], 16) % len(templates)]


def build_records(pairs: list[dict], *, limit: int, min_words: int, both_directions: bool) -> list[dict]:
    records: list[dict] = []
    seen: set[str] = set()
    for pair in pairs:
        if len(records) >= limit:
            break
        luganda, english = pair["luganda"].strip(), pair["english"].strip()
        if len(luganda.split()) < min_words or len(english.split()) < min_words:
            continue                                     # one-word verses teach nothing
        key = luganda.lower()
        if key in seen:
            continue
        seen.add(key)
        ref = pair.get("ref", "")
        records.append({
            "messages": [
                {"role": "user", "content": pick_template(ref, TO_LUGANDA).format(source=english)},
                {"role": "assistant", "content": luganda},
            ],
            "topic": "bible-translation",
            "register": "biblical",
            "source": "bible-parallel",
            "ref": ref,
            "verified": True,
        })
        if both_directions and len(records) < limit:
            records.append({
                "messages": [
                    {"role": "user", "content": pick_template(ref + "-en", TO_ENGLISH).format(source=luganda)},
                    {"role": "assistant", "content": english},
                ],
                "topic": "bible-translation-en",
                "register": "biblical",
                "source": "bible-parallel",
                "ref": ref,
                "verified": True,
            })
    return records


def allowed_count(existing: int, share: float, hard_limit: int) -> tuple[int, str]:
    """How many translation records may join a corpus of ``existing`` chat records."""
    if existing <= 0:
        return hard_limit, ("no other SFT records on disk — using the hard limit; the mix is "
                           "100% translation until you add seed conversations")
    budget = int(share / (1 - share) * existing)
    if budget <= 0:
        return 0, f"a {share:.0%} share of {existing} record(s) rounds to zero"
    if budget < hard_limit:
        return budget, f"capped at {share:.0%} of {existing} existing SFT record(s)"
    return hard_limit, f"under the {share:.0%} cap ({budget} allowed)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default=str(DEFAULT_PAIRS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--limit", type=int, default=4000, help="hard ceiling on new records")
    ap.add_argument("--max-share", type=float, default=0.20,
                    help="most this dataset may add, as a share of the existing SFT set")
    ap.add_argument("--min-words", type=int, default=4, help="skip verses shorter than this")
    ap.add_argument("--both-directions", action="store_true", help="also LG->EN records")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    pairs = load_pairs(Path(args.pairs))
    existing = existing_sft_records(out)
    limit, reason = allowed_count(existing, args.max_share, args.limit)
    records = build_records(pairs, limit=limit, min_words=args.min_words,
                            both_directions=args.both_directions)

    words = sum(len(m["content"].split()) for r in records for m in r["messages"])
    print(f"pairs available     {len(pairs):,}")
    print(f"existing SFT set    {existing:,} record(s)")
    print(f"allowed             {limit:,} — {reason}")
    print(f"would write         {len(records):,} record(s), {words:,} words")
    if not records:
        print("\nNothing to write. Either the cap is zero (add conversation records first) or "
              "the pairs file is empty/short.")
        return 0 if args.dry_run else 1
    if args.dry_run:
        for rec in records[:3]:
            print(f"\n  user: {rec['messages'][0]['content']}")
            print(f"  asst: {rec['messages'][1]['content'][:120]}")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nwrote {out}")
    print("Next: python3 finetune/validate_dataset.py finetune/data   (then build_dataset.py)")
    print("Remember what this is: translation pairs teach meaning, not conversation. Keep the "
          "share small and keep letters/chat in the mix.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
