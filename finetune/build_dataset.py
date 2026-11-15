#!/usr/bin/env python3
"""Merge every Luganda source into clean, split, training-ready files.

Inputs (only the ones that exist are used):
    finetune/data/seed_luganda.jsonl      hand-written seed conversations
    finetune/data/ingested/*.jsonl        output of ingest_hf.py
    finetune/data/exported/*.jsonl        output of the app's Export button

Outputs, in finetune/out/:
    train.jsonl  val.jsonl  test.jsonl    OpenAI chat format ({"messages": [...]})
                                          — also what TRL/peft expect
    cpt.txt                               optional raw-text stream for continued
                                          pre-training on Luganda prose
    stats.json  manifest.json             what went in, what came out, and why

Usage:
    python finetune/build_dataset.py
    python finetune/build_dataset.py --verified-only --cpt
    python finetune/build_dataset.py --include finetune/data/my_extra.jsonl --val 0.1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "out"
DEFAULT_SOURCES = [DATA / "seed_luganda.jsonl"]
GLOB_SOURCES = [DATA / "ingested", DATA / "exported"]


# --------------------------------------------------------------------------- #
def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def key_of(record: dict) -> str:
    """Stable de-dupe key: the whole thread, case/punctuation folded."""
    blob = " ".join(m.get("content", "") for m in record["messages"])
    blob = re.sub(r"[^a-z0-9 ]", "", blob.lower())
    return hashlib.sha1(blob.encode()).hexdigest()


def prompt_key(record: dict) -> str:
    first = next((m["content"] for m in record["messages"] if m["role"] == "user"), "")
    return re.sub(r"[^a-z0-9 ]", "", first.lower()).strip()


def bucket(key: str, val: float, test: float) -> str:
    """Deterministic split: the same record always lands in the same split."""
    x = int(key[:8], 16) / 0xFFFFFFFF
    if x < test:
        return "test"
    if x < test + val:
        return "val"
    return "train"


def load(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "messages" in rec:
            out.append(rec)
    return out


def clean_record(rec: dict, *, min_chars: int, max_turns: int) -> tuple[dict | None, str]:
    msgs = []
    for m in rec.get("messages", []):
        role, content = m.get("role"), normalize(m.get("content", ""))
        if role not in {"system", "user", "assistant"} or not content:
            return None, "bad-turn"
        msgs.append({"role": role, "content": content})
    if not msgs or msgs[-1]["role"] != "assistant":
        return None, "no-assistant-target"
    if not any(m["role"] == "user" for m in msgs):
        return None, "no-user-turn"
    if len(msgs) > max_turns:
        return None, "too-long"
    if any(len(m["content"]) < min_chars for m in msgs):
        return None, "too-short"
    return {
        "messages": msgs,
        "source": rec.get("source", "unknown"),
        "verified": bool(rec.get("verified", False)),
        "topic": rec.get("topic", ""),
    }, "ok"


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps({"messages": rec["messages"]}, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--include", nargs="*", default=[], help="extra .jsonl files or directories")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--val", type=float, default=0.1, help="validation fraction")
    ap.add_argument("--test", type=float, default=0.1, help="test fraction")
    ap.add_argument("--verified-only", action="store_true", help="keep only native-verified records")
    ap.add_argument("--min-chars", type=int, default=2)
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--cpt", action="store_true", help="also write cpt.txt for continued pre-training")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = list(DEFAULT_SOURCES)
    for folder in GLOB_SOURCES:
        if folder.exists():
            sources.extend(sorted(folder.glob("*.jsonl")))
    for extra in args.include:
        p = Path(extra)
        sources.extend(sorted(p.rglob("*.jsonl")) if p.is_dir() else [p])

    raw: list[dict] = []
    per_source: Counter = Counter()
    for path in sources:
        recs = load(path)
        for rec in recs:
            rec.setdefault("source", path.stem)
            raw.append(rec)
        per_source[path.name] = len(recs)
        print(f"  read {len(recs):5d}  {path}")

    kept: list[dict] = []
    drops: Counter = Counter()
    seen: dict[str, dict] = {}
    dups = 0
    for rec in raw:
        clean, why = clean_record(rec, min_chars=args.min_chars, max_turns=args.max_turns)
        if not clean:
            drops[why] += 1
            continue
        if args.verified_only and not clean["verified"]:
            drops["unverified"] += 1
            continue
        k = key_of(clean)
        if k in seen:
            dups += 1
            continue          # keep the first occurrence (seed files sort first)
        seen[k] = clean
        kept.append(clean)

    splits = {"train": [], "val": [], "test": []}
    for rec in kept:
        splits[bucket(key_of(rec), args.val, args.test)].append(rec)

    # Leakage guard: a val/test prompt that also appears in train is worthless.
    train_prompts = {prompt_key(r) for r in splits["train"]}
    leaked = 0
    for name in ("val", "test"):
        before = len(splits[name])
        splits[name] = [r for r in splits[name] if prompt_key(r) not in train_prompts]
        leaked += before - len(splits[name])

    # A tiny corpus can produce an empty split; move one record rather than ship nothing.
    for name in ("val", "test"):
        if not splits[name] and len(splits["train"]) > 3:
            splits[name].append(splits["train"].pop())

    for name, records in splits.items():
        write_jsonl(out_dir / f"{name}.jsonl", records)

    if args.cpt:
        with (out_dir / "cpt.txt").open("w", encoding="utf-8") as fh:
            for rec in kept:
                for m in rec["messages"]:
                    fh.write(m["content"] + "\n\n")
        # NOTE: cpt.txt must contain *monolingual natural text*, not chat turns,
        # for continued pre-training. Prefer luganda_prose for this.

    all_recs = [r for rs in splits.values() for r in rs]
    turn_lens = [len(m["content"].split()) for r in all_recs for m in r["messages"] if m["role"] == "assistant"]
    tokens = sum(len(m["content"].split()) for r in all_recs for m in r["messages"])
    vocab = Counter(w.lower() for r in all_recs for m in r["messages"] for w in re.findall(r"[a-zA-Z']+", m["content"]))
    stats = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "records": {"train": len(splits["train"]), "val": len(splits["val"]), "test": len(splits["test"]), "total": len(all_recs)},
        "words_total": tokens,
        "words_assistant_mean": round(statistics.fmean(turn_lens), 1) if turn_lens else 0,
        "words_assistant_p95": sorted(turn_lens)[int(len(turn_lens) * 0.95)] if turn_lens else 0,
        "vocabulary_size": len(vocab),
        "duplicates_removed": dups,
        "drops": dict(drops),
        "split_leakage_removed": leaked,
        "verified_fraction": round(sum(r["verified"] for r in all_recs) / max(len(all_recs), 1), 3),
        "sources": dict(per_source),
        "top_words": vocab.most_common(25),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "manifest.json").write_text(
        json.dumps({"sources": [str(s) for s in sources], "args": vars(args)}, indent=2), encoding="utf-8"
    )

    print(json.dumps({k: stats[k] for k in ("records", "words_total", "duplicates_removed", "drops",
                                            "split_leakage_removed", "verified_fraction")}, indent=2))
    if stats["records"]["total"] < 50:
        print("\n! Only %d training records. Enough to smoke-test the pipeline, NOT enough to teach a"
              " model Luganda. Add a corpus (see finetune/README.md) before you train for real."
              % stats["records"]["total"])
    print(f"\nwrote {out_dir}/{'train,val,test'}.jsonl + stats.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
