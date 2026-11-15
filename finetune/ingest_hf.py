#!/usr/bin/env python3
"""Pull public Luganda corpora from the Hugging Face Hub into this pipeline.

Three transport strategies, tried in order (override with --strategy):
  parquet  Hub parquet export (fast, needs pyarrow)
  raw      the repo's .jsonl/.json/.csv files (stdlib only)
  rows     public datasets-server REST API (no deps, but not every repo is indexed)

Presets:
  alpaca   instruction/input/output rows  -> SFT chat records
  salt     parallel English/Luganda rows  -> translation SFT, both directions
  chat     messages/conversations rows    -> SFT chat records
  prose    monolingual text               -> data/ingested/mono/*.txt (continued
                                             pre-training, NOT chat fine-tuning)
  auto     guess from the row fields

Examples:
  python finetune/ingest_hf.py --dataset allandclive/alpaca_4k_luganda --preset alpaca
  python finetune/ingest_hf.py --dataset Sunbird/salt --preset salt --config text --limit 20000
  python finetune/ingest_hf.py --dataset kambale/luganda_prose --preset prose --limit 200000

Check each dataset's licence before training (SALT is CC-BY-SA-4.0, most
alpaca-style Luganda sets are Apache/MIT, prose corpora vary).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator

import httpx

DATA = Path(__file__).resolve().parent / "data"
HUB = "https://huggingface.co"
API = f"{HUB}/api/datasets"
SERVER = "https://datasets-server.huggingface.co"
PAGE = 100


# --------------------------------------------------------------------------- #
# Row mappers
# --------------------------------------------------------------------------- #
def pick(row: dict, *names: str) -> str | None:
    lowered = {str(k).lower(): v for k, v in row.items()}
    for name in names:
        value = lowered.get(name.lower())
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def map_alpaca(row: dict) -> dict | None:
    instruction = pick(row, "instruction", "question", "prompt", "input_text")
    output = pick(row, "output", "answer", "response", "completion", "luganda", "translation")
    extra = pick(row, "input", "context")
    if not instruction or not output:
        return None
    user = f"{instruction}\n{extra}" if extra else instruction
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": output}]}


def map_salt(row: dict) -> list[dict]:
    lug = pick(row, "luganda", "lug", "ganda")
    eng = pick(row, "english", "eng", "en")
    if not lug or not eng:
        return []
    return [
        {"messages": [{"role": "user", "content": f'Kyusa mu Luganda: "{eng}"'},
                      {"role": "assistant", "content": lug}]},
        {"messages": [{"role": "user", "content": f'Translate into English: "{lug}"'},
                      {"role": "assistant", "content": eng}]},
    ]


def map_chat(row: dict) -> dict | None:
    raw = row.get("messages") or row.get("conversations")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if not isinstance(raw, list) or len(raw) < 2:
        return None
    msgs = []
    for m in raw:
        if not isinstance(m, dict):
            return None
        role = str(m.get("role") or m.get("from") or "").lower()
        content = str(m.get("content") or m.get("value") or "").strip()
        if content and role in {"human", "user", "gpt", "assistant", "bot"}:
            msgs.append({"role": "user" if role in {"human", "user"} else "assistant", "content": content})
    if len(msgs) >= 2 and msgs[-1]["role"] == "assistant" and msgs[0]["role"] == "user":
        return {"messages": msgs}
    return None


def guess_preset(row: dict) -> str:
    keys = {str(k).lower() for k in row}
    if {"instruction", "output"} <= keys or {"question", "answer"} <= keys:
        return "alpaca"
    if {"luganda", "english"} <= keys or ({"lug"} & keys and {"eng"} & keys):
        return "salt"
    if {"messages", "conversations"} & keys:
        return "chat"
    return "prose"


# --------------------------------------------------------------------------- #
# Transports: each yields dict rows
# --------------------------------------------------------------------------- #
def rows_parquet(client: httpx.Client, dataset: str, config: str, split: str, limit: int) -> Iterator[dict]:
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415
    except ImportError:
        raise SystemExit("pyarrow is required for --strategy parquet: pip install pyarrow")
    listing = client.get(f"{API}/{dataset}/parquet", timeout=60).json()
    if not listing or "error" in listing:
        raise RuntimeError(f"no parquet export available ({listing.get('error', 'empty response') if isinstance(listing, dict) else 'empty'})")
    configs = list(listing)
    if config in listing:
        chosen = config
    else:
        preferred = [c for c in configs if "text" in c.lower()] or configs
        chosen = preferred[0]
    node = listing[chosen]
    if isinstance(node, list):           # flat shape: config -> [urls]
        files = node
    else:
        files = node.get(split) or next(iter(node.values()))
    print(f"  parquet config={chosen} split={split} files={len(files)}")
    emitted = 0
    for url in files:
        if emitted >= limit:
            return
        with client.stream("GET", url, timeout=300, follow_redirects=True) as resp:
            resp.raise_for_status()
            buf = io.BytesIO(resp.read())
        pf = pq.ParquetFile(buf)
        for batch in pf.iter_batches(batch_size=512):
            for row in batch.to_pylist():
                yield row
                emitted += 1
                if emitted >= limit:
                    return


def rows_raw(client: httpx.Client, dataset: str, limit: int) -> Iterator[dict]:
    tree = client.get(f"{API}/{dataset}/tree/main", params={"recursive": "true"}, timeout=60)
    tree.raise_for_status()
    entries = tree.json()
    if isinstance(entries, dict):
        entries = entries.get("siblings", [])
    candidates = [
        e["path"] for e in entries
        if str(e.get("path", "")).endswith((".jsonl", ".json", ".csv"))
        and e.get("size", 1) < 400 * 1024 * 1024
    ]
    if not candidates:
        raise RuntimeError("no json/jsonl/csv files in the repository root")
    emitted = 0
    for path in candidates:
        if emitted >= limit:
            return
        url = f"{HUB}/datasets/{dataset}/resolve/main/{path}"
        body = client.get(url, timeout=300, follow_redirects=True).text
        for row in _parse_text_file(path, body):
            yield row
            emitted += 1
            if emitted >= limit:
                return


def _parse_text_file(path: str, body: str) -> Iterator[dict]:
    if path.endswith(".jsonl"):
        for line in body.splitlines():
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
        return
    if path.endswith(".csv"):
        import csv

        for row in csv.DictReader(io.StringIO(body)):
            yield dict(row)
        return
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return
    if isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, list):
                payload = value
                break
    if isinstance(payload, list):
        for row in payload:
            if isinstance(row, dict):
                yield row


def rows_server(client: httpx.Client, dataset: str, config: str, split: str, limit: int, offset: int) -> Iterator[dict]:
    if not config or not split:
        info = client.get(f"{SERVER}/splits", params={"dataset": dataset}, timeout=60).json()
        splits = info.get("splits") or []
        if not splits:
            raise RuntimeError("datasets-server has no splits for this repo")
        chosen = next((s for s in splits if s["split"] == "train"), splits[0])
        config, split = chosen["config"], chosen["split"]
    seen = 0
    while seen < limit:
        length = min(PAGE, limit - seen)
        resp = client.get(
            f"{SERVER}/rows",
            params={"dataset": dataset, "config": config, "split": split, "offset": offset + seen, "length": length},
            timeout=90,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"datasets-server HTTP {resp.status_code}")
        batch = (resp.json().get("rows") or [])
        if not batch:
            return
        for item in batch:
            yield item["row"]
        seen += len(batch)
        if len(batch) < length:
            return
        time.sleep(0.15)


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--preset", default="auto", choices=["auto", "alpaca", "salt", "chat", "prose"])
    ap.add_argument("--strategy", default="auto", choices=["auto", "parquet", "raw", "rows"])
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--config", default="")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    slug = args.dataset.replace("/", "__")
    (DATA / "ingested").mkdir(parents=True, exist_ok=True)
    (DATA / "ingested" / "mono").mkdir(parents=True, exist_ok=True)

    attempts = [args.strategy] if args.strategy != "auto" else ["parquet", "raw", "rows"]
    stream: Iterable[dict] | None = None
    errors = []
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or ""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with httpx.Client(follow_redirects=True, headers=headers) as client:
        for strategy in attempts:
            try:
                if strategy == "parquet":
                    stream = rows_parquet(client, args.dataset, args.config, args.split, args.limit)
                    first = next(iter(stream))
                    stream = _prepend(first, stream)
                elif strategy == "raw":
                    stream = rows_raw(client, args.dataset, args.limit)
                    first = next(iter(stream))
                    stream = _prepend(first, stream)
                else:
                    stream = rows_server(client, args.dataset, args.config, args.split, args.limit, args.offset)
                    first = next(iter(stream))
                    stream = _prepend(first, stream)
                print(f"  strategy: {strategy}")
                break
            except StopIteration:
                errors.append(f"{strategy}: no rows")
                stream = None
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in (401, 403):
                    errors.append(f"{strategy}: gated/private dataset (HTTP {exc.response.status_code})")
                else:
                    errors.append(f"{strategy}: HTTP {exc.response.status_code}")
                stream = None
            except (RuntimeError, httpx.HTTPError, SystemExit) as exc:
                errors.append(f"{strategy}: {type(exc).__name__} {exc}")
                stream = None
        if stream is None:
            print("Nothing worked for this dataset:")
            for err in errors:
                print("  -", err)
            print(f"Inspect it with: curl '{API}/{args.dataset}/tree/main?recursive=true'")
            print("If a dataset is gated, accept its terms on the Hub then set HF_TOKEN=hf_... and re-run.")
            return 1

        records: list[dict] = []
        mono: list[str] = []
        counts = {"rows": 0, "records": 0, "skipped": 0}
        for row in stream:
            counts["rows"] += 1
            use = guess_preset(row) if args.preset == "auto" else args.preset
            if use == "prose":
                text = pick(row, "text", "content", "sentence", "body")
                if text and len(text.split()) >= 5:
                    mono.append(text)
                else:
                    counts["skipped"] += 1
                counts["records"] = len(records)
                if sum(len(t.split()) for t in mono) > 5_000_000:
                    break
                continue
            if use == "salt":
                records.extend(map_salt(row))
                continue
            rec = map_chat(row) if use == "chat" else map_alpaca(row)
            if rec:
                records.append(rec)
            else:
                counts["skipped"] += 1

    if records:
        out = Path(args.out) if args.out else DATA / "ingested" / f"{slug}.jsonl"
        with out.open("w", encoding="utf-8") as fh:
            for rec in records:
                rec.update({"source": f"hf:{args.dataset}", "verified": False,
                            "topic": "translation" if args.preset == "salt" else "instruction"})
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        counts["records"] = len(records)
        print(f"  wrote {len(records)} chat records -> {out}")

    if mono:
        out = DATA / "ingested" / "mono" / f"{slug}.txt"
        out.write_text("\n".join(mono), encoding="utf-8")
        print(f"  wrote {len(mono)} lines -> {out} ({sum(len(t.split()) for t in mono):,} words)")

    print(json.dumps(counts))
    return 0 if (records or mono) else 1


def _prepend(first: dict, rest: Iterable[dict]) -> Iterator[dict]:
    yield first
    yield from rest


if __name__ == "__main__":
    sys.exit(main())
