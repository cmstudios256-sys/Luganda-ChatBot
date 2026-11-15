#!/usr/bin/env python3
"""Score a Luganda model on the held-out eval set.

Three layers, from cheapest to most meaningful:
  1. chrF++ and BLEU against reference answers      (automatic, noisy, useful for regressions)
  2. an LLM judge on a 5-point rubric               (correlates with humans, not a substitute)
  3. a native-speaker rubric sheet you actually print and fill in

Automatic metrics do not measure "does this sound like a Muganda". They measure
"does this look like the reference". Use them to catch regressions between
checkpoints, never to declare victory.

Usage:
    python finetune/eval.py --predictions preds.jsonl
    python finetune/eval.py                       # generate predictions with the app's provider
    python finetune/eval.py --judge                # also run the LLM judge (needs an API key)

Prediction file: one JSON object per line, {"id": ..., "prediction": "..."}.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_EVAL = ROOT / "data" / "eval" / "lug_eval_set.jsonl"
OUT_DIR = ROOT / "out"


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
from eval_lib import bleu, chrf_plus_plus  # noqa: E402

def judge_prompt(item: dict, prediction: str) -> str:
    return (
        "You are a strict Luganda native-speaker examiner. Score the CANDIDATE reply.\n"
        "Return ONLY JSON with integer 1-5 scores and no prose:\n"
        '{"fluency":n,"grammar":n,"naturalness":n,"culture_fit":n,"language_purity":n,'
        '"reason":"<=25 words in English"}\n\n'
        "Scoring guide:\n"
        "5 = a native speaker could have written it; 3 = understandable but clearly non-native;"
        " 1 = not Luganda / hallucinated words / ignores the question.\n"
        "language_purity: 5 = no unnecessary English, 1 = mostly English.\n"
        "Be harsh. Do not reward effort.\n\n"
        f"PROMPT: {item['prompt']}\n"
        f"REFERENCE ANSWER (may itself be imperfect): {item['reference']}\n"
        f"CANDIDATE: {prediction}"
    )


async def run_judge(items: list[dict], preds: dict[str, str]) -> list[dict]:
    sys.path.insert(0, str(ROOT.parent))
    from app.config import SETTINGS  # noqa: PLC0415
    from app.providers import get_provider  # noqa: PLC0415

    if not SETTINGS.is_live:
        print("! no API key configured — skipping the LLM judge (set OPENAI_API_KEY or similar)")
        return []
    provider = get_provider()
    results = []
    for item in items:
        pred = preds.get(item["id"], "")
        if not pred:
            continue
        raw = await provider.complete(
            [{"role": "user", "content": judge_prompt(item, pred)}],
            system="You output only valid JSON. No markdown, no commentary.",
            temperature=0.0,
            max_tokens=200,
        )
        match = re.search(r"\{.*\}", raw, re.S)
        try:
            scores = json.loads(match.group(0)) if match else {}
        except json.JSONDecodeError:
            scores = {}
        results.append({"id": item["id"], "scores": scores, "raw": raw[:400]})
    return results


# --------------------------------------------------------------------------- #
# Prediction generation
# --------------------------------------------------------------------------- #
async def generate_predictions(items: list[dict]) -> dict[str, str]:
    sys.path.insert(0, str(ROOT.parent))
    from app.config import SETTINGS  # noqa: PLC0415
    from app.prompts import build_system_prompt  # noqa: PLC0415
    from app.providers import get_provider  # noqa: PLC0415

    if not SETTINGS.is_live:
        raise SystemExit(
            "No API key configured, so there is nothing to generate predictions with.\n"
            "Either put a key in .env, or point --predictions at a file of model outputs."
        )
    provider = get_provider()
    system = build_system_prompt()
    preds = {}
    for item in items:
        preds[item["id"]] = (
            await provider.complete(
                [{"role": "user", "content": item["prompt"]}], system=system, temperature=0.2, max_tokens=400
            )
        ).strip()
        print(f"  {item['id']:>18}  {preds[item['id']][:70]}")
    return preds


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-set", default=str(DEFAULT_EVAL))
    ap.add_argument("--predictions", default="", help="jsonl with {id, prediction}")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()

    eval_path = Path(args.eval_set)
    if not eval_path.exists():
        print(f"eval set not found: {eval_path}")
        return 2
    items = [json.loads(l) for l in eval_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    if args.predictions:
        preds = {}
        for line in Path(args.predictions).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                preds[str(rec["id"])] = rec.get("prediction", "")
    else:
        preds = asyncio.run(generate_predictions(items))

    scored = [(it, preds[it["id"]], it["reference"]) for it in items if preds.get(it["id"])]
    if not scored:
        print("No overlapping predictions and eval items — check the ids.")
        return 1

    per_item = []
    for item, pred, ref in scored:
        per_item.append({
            "id": item["id"], "category": item.get("category", ""), "prompt": item["prompt"],
            "prediction": pred, "reference": ref,
            "chrf": round(chrf_plus_plus(pred, ref), 2),
            "rubric_note": item.get("rubric_note", ""),
        })

    corpus_chrf = statistics.fmean(p["chrf"] for p in per_item)
    by_cat: dict[str, list[float]] = {}
    for p in per_item:
        by_cat.setdefault(p["category"], []).append(p["chrf"])
    bleu_scores = bleu([(p["prediction"], p["reference"]) for p in per_item])

    judge = asyncio.run(run_judge(items, preds)) if args.judge else []
    judge_summary = {}
    if judge:
        dims = ["fluency", "grammar", "naturalness", "culture_fit", "language_purity"]
        for dim in dims:
            vals = [j["scores"][dim] for j in judge if isinstance(j["scores"].get(dim), (int, float))]
            if vals:
                judge_summary[dim] = round(statistics.fmean(vals), 2)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": os.getenv("LLM_MODEL", ""),
        "provider": os.getenv("LLM_PROVIDER", "auto"),
        "n_items": len(per_item),
        "chrf_plus_plus_mean": round(corpus_chrf, 2),
        "chrf_by_category": {k: round(statistics.fmean(v), 2) for k, v in sorted(by_cat.items())},
        **bleu_scores,
        "judge": judge_summary or "skipped",
        "items": per_item,
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Native-speaker sheet: the number that actually decides whether you ship.
    with (out_dir / "human_rubric.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "category", "prompt", "model_answer", "reference",
                         "native_1_5_fluency", "native_1_5_naturalness", "native_1_5_grammar",
                         "native_1_5_culture", "would_a_muganda_say_this_y_n", "native_comment"])
        for p in per_item:
            writer.writerow([p["id"], p["category"], p["prompt"], p["prediction"], p["reference"], "", "", "", "", "", ""])

    print(json.dumps({k: report[k] for k in
                      ("n_items", "chrf_plus_plus_mean", "chrf_by_category", "bleu", "judge")}, indent=2))
    print(f"\nwrote {out_dir/'eval_report.json'} and {out_dir/'human_rubric.csv'}")
    print("Next: get three native speakers to fill human_rubric.csv. If 'would a Muganda say this' is")
    print("below 80% yes, fix the data, not the hyperparameters.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
