#!/usr/bin/env python3
"""Fine-tune without a GPU: hand the dataset to a hosted fine-tuning API.

Your machine only uploads a JSONL file and polls a job. The provider owns the
GPUs. Same data the local QLoRA path uses (`finetune/out/train.jsonl`), because
OpenAI's chat format is exactly the `{"messages": [...]}` shape we already emit.

Usage:
    python finetune/train_openai.py --dry-run                      # validate the files, no network
    python finetune/train_openai.py                                # upload + start the job
    python finetune/train_openai.py --status ftjob-abc123          # check on it
    python finetune/train_openai.py --list                         # recent jobs
    python finetune/train_openai.py --base-url https://api.together.xyz/v1   # any compatible host

Cost, honestly: a few dollars for a few thousand short conversations on the
"mini" model classes. Check the provider's current price page before you start —
this script prints the token count so you can do that arithmetic yourself.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN = ROOT / "out" / "train.jsonl"
DEFAULT_VAL = ROOT / "out" / "val.jsonl"
MAX_UPLOAD_BYTES = 500 * 1024 * 1024   # OpenAI's documented file limit


def count_tokens_estimate(path: Path) -> tuple[int, int]:
    """(examples, rough tokens). Luganda is ~1.5-1.7 tokens/word on modern BPE."""
    examples = words = 0
    if not path.exists():
        return 0, 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        examples += 1
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        for msg in rec.get("messages", []):
            words += len(str(msg.get("content", "")).split())
    return examples, int(words * 1.6)


def validate(path: Path, *, minimum: int) -> list[str]:
    """OpenAI's hard requirements, checked locally so you do not pay for a failed upload."""
    problems: list[str] = []
    if not path.exists():
        return [f"{path} does not exist — run finetune/build_dataset.py first"]
    size = path.stat().st_size
    if size > MAX_UPLOAD_BYTES:
        problems.append(f"{path.name}: {size/1e6:.0f} MB exceeds the 500 MB upload limit; split it")
    examples = 0
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        examples += 1
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            problems.append(f"{path.name}:{i} is not valid JSON")
            continue
        msgs = rec.get("messages")
        if not isinstance(msgs, list) or len(msgs) < 2:
            problems.append(f"{path.name}:{i} needs at least one user and one assistant message")
            continue
        if msgs[0].get("role") == "system":
            problems.append(f"{path.name}:{i} starts with a system message — move those rules into the app prompt")
        if msgs[-1].get("role") != "assistant":
            problems.append(f"{path.name}:{i} must end with an assistant message")
        for m in msgs:
            if m.get("role") not in {"system", "user", "assistant"}:
                problems.append(f"{path.name}:{i} has an unknown role {m.get('role')!r}")
            if not str(m.get("content", "")).strip():
                problems.append(f"{path.name}:{i} has an empty message")
        if len(problems) > 25:
            problems.append("… stopping after 25 problems")
            break
    if examples < minimum:
        problems.append(
            f"{path.name}: only {examples} examples. Hosted fine-tuning needs {minimum}+ to do"
            " anything useful; below that the prompt+retrieval path will beat it."
        )
    return problems


def api(base_url: str, key: str) -> httpx.Client:
    return httpx.Client(
        base_url=base_url.rstrip("/"),
        headers={"Authorization": f"Bearer {key}"},
        timeout=300.0,
    )


def upload(client: httpx.Client, path: Path) -> str:
    print(f"  uploading {path.name} ({path.stat().st_size/1e3:.0f} KB)…")
    with path.open("rb") as fh:
        r = client.post("/files", files={"file": (path.name, fh, "application/jsonl")},
                        data={"purpose": "fine-tune"})
    r.raise_for_status()
    file_id = r.json()["id"]
    print(f"  file id: {file_id}")
    return file_id


def create_job(client: httpx.Client, *, train_id: str, val_id: str, model: str,
               epochs: int, suffix: str) -> str:
    payload = {
        "training_file": train_id,
        "model": model,
        "suffix": suffix,
        "hyperparameters": {"n_epochs": epochs},
    }
    if val_id:
        payload["validation_file"] = val_id
    r = client.post("/fine_tuning/jobs", json=payload)
    if r.status_code >= 400:
        raise SystemExit(f"fine-tuning job failed: HTTP {r.status_code} {r.text[:400]}")
    job = r.json()
    print(f"  job id: {job['id']}  status: {job.get('status')}")
    return job["id"]


def poll(client: httpx.Client, job_id: str, *, once: bool = False) -> None:
    while True:
        r = client.get(f"/fine_tuning/jobs/{job_id}")
        r.raise_for_status()
        job = r.json()
        status = job.get("status", "?")
        line = f"  {status}"
        if job.get("trained_tokens"):
            line += f"  trained_tokens={job['trained_tokens']:,}"
        if job.get("fine_tuned_model"):
            line += f"  model={job['fine_tuned_model']}"
        print(line)
        if status in {"succeeded", "failed", "cancelled"}:
            if status == "succeeded":
                print(f"\nUse it:  LLM_PROVIDER=openai  LLM_MODEL={job.get('fine_tuned_model')}")
                print("Then re-run finetune/eval.py and compare chrF++ with the base model.")
            elif job.get("error"):
                print(f"  error: {job['error']}")
            return
        if once:
            return
        time.sleep(30)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default=str(DEFAULT_TRAIN))
    ap.add_argument("--val", default=str(DEFAULT_VAL))
    ap.add_argument("--model", default=os.getenv("FINETUNE_BASE_MODEL", "gpt-4.1-mini-2025-04-14"),
                    help="a model the provider allows fine-tuning on; check their docs")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--suffix", default="yogera-luganda")
    ap.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    ap.add_argument("--minimum-examples", type=int, default=100)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", default="")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    train_path, val_path = Path(args.train), Path(args.val)
    examples, tokens = count_tokens_estimate(train_path)
    print(f"train: {train_path}  examples={examples}  ~tokens={tokens:,}")

    problems = validate(train_path, minimum=args.minimum_examples)
    if val_path.exists():
        problems += [p for p in validate(val_path, minimum=10) if "at least" not in p]
    else:
        print("  (no validation file — training will still run, but you lose the eval curve)")

    if problems:
        print("\nProblems:")
        for p in problems:
            print("  -", p)
        if args.dry_run:
            return 1
        print("\nRefusing to upload a dataset that will fail or waste money. Fix these first.")
        return 1

    if args.dry_run:
        print("\nDry run OK — the files satisfy the provider's format rules.")
        print(f"Rough training cost: {tokens:,} tokens x {args.epochs} epochs. "
              "Multiply by the provider's per-token training price.")
        return 0

    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        print("No OPENAI_API_KEY in the environment (or .env). Add one, or use --dry-run.")
        return 2

    with api(args.base_url, key) as client:
        if args.list:
            r = client.get("/fine_tuning/jobs", params={"limit": 20})
            r.raise_for_status()
            for job in r.json().get("data", []):
                print(f"  {job['id']:<28} {job.get('status'):<12} {job.get('fine_tuned_model') or ''}")
            return 0
        if args.status:
            poll(client, args.status, once=True)
            return 0
        train_id = upload(client, train_path)
        val_id = upload(client, val_path) if val_path.exists() else ""
        job_id = create_job(client, train_id=train_id, val_id=val_id, model=args.model,
                            epochs=args.epochs, suffix=args.suffix)
    print(f"\nPolling {job_id} (Ctrl-C is safe; poll later with --status {job_id})")
    with api(args.base_url, key) as client:
        poll(client, job_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
