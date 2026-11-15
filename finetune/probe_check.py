#!/usr/bin/env python3
"""Run the conversation probes against a live bot and score the checks that need no human.

    python3 -m finetune.probe_check --dry-run                 # read the probe set
    python3 -m finetune.probe_check --url http://localhost:8000
    python3 -m finetune.probe_check --url … --json out.json

This is deliberately *not* a quality judge. A machine cannot tell you whether a reply sounds
like a Muganda — three native speakers with the rubric in VOICE-PLAN.md do that. What a
machine can do, cheaply and repeatably, is catch the failures humans get tired of reporting:

* English leaking into a Luganda reply
* a clock written as ``7:00 AM`` instead of ``ssaawa emu ey'oku makya``
* a wall of text when one sentence was asked for
* "definitely" answers to questions the bot cannot know (prices, weather)
* ignoring an explicit must-appear / must-not-appear instruction in the probe

Exit code is 1 when any hard check fails, so it drops straight into CI or a make target.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROBES = Path(__file__).resolve().parent / "probes" / "conversation_probes.jsonl"

CLOCK_24 = re.compile(r"\b\d{1,2}:\d{2}\b")
AM_PM = re.compile(r"\b\d{1,2}\s?(?:a\.?m\.?|p\.?m\.?)\b|\b(?:AM|PM)\b")
# words that are fine inside Luganda sentences (names, borrowed-but-standard terms)
ENGLISH_ALLOW = {
    "yogera", "whatsapp", "facebook", "google", "jinja", "kampala", "uganda", "entebbe",
    "mukono", "kiyira", "nile", "ssemakula", "kato", "wasswa", "musoke", "mukasa",
    "ok", "sms", "tv", "radio", "boda", "taxi", "bus", "ssomero",
}
# a reply that claims certainty about things it cannot know
UNKNOWABLE = ("ssente", "price", "obudde", "weather", "bw'olunaku", "leero")
CERTAINTY = ("ddala", "kakati", "mazima", "definitely", "exactly", "cert")


def load_probes(path: Path = PROBES) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# English words that show up inside otherwise-Luganda replies
ENGLISH_STOP = {
    "the", "and", "you", "your", "for", "with", "that", "this", "these", "those", "have",
    "will", "would", "can", "could", "are", "was", "were", "but", "not", "from", "about",
    "here", "there", "what", "when", "which", "how", "why", "please", "sorry", "hello",
    "thanks", "however", "because", "also", "just", "very", "settings", "api", "model",
}
# letter sequences Luganda does not use — cheap, and far more precise than vowel counting
ENGLISH_MARKERS = ("th", "wh", "ck", "qu", "tion", "ing", "ould", "ight")


def looks_english(word: str) -> bool:
    """Is this token English rather than Luganda?

    An earlier version used only a vowel-ratio rule and flagged real Luganda ("gwange",
    "ndyoke") because a short Luganda word can be consonant-heavy. Markers plus a stopword
    list are cruder but they do not accuse the language of being English.
    """
    low = word.lower()
    if low in ENGLISH_ALLOW:
        return False
    if word.isupper() and len(word) >= 2:          # API, URL, PDF — technical, but not Luganda
        return True
    if low in ENGLISH_STOP:
        return True
    vowels = sum(1 for ch in low if ch in "aeiou")
    if any(m in low for m in ENGLISH_MARKERS) and vowels:
        return True
    return len(low) >= 5 and vowels / len(low) < 0.25


def english_leak(reply: str) -> list[str]:
    """Tokens in the reply that are not Luganda-looking."""
    return [w for w in re.findall(r"[A-Za-z][A-Za-z'’-]{2,}", reply) if looks_english(w)]


def check_reply(probe: dict, reply: str) -> list[dict]:
    """Return a list of {level, check, detail}. level is 'fail' or 'warn'."""
    problems: list[dict] = []
    low = reply.lower()
    words = len(reply.split())

    if not reply.strip():
        return [{"level": "fail", "check": "empty", "detail": "no reply at all"}]

    leaks = english_leak(reply)
    if len(leaks) >= 3:
        problems.append({"level": "fail", "check": "english-leak",
                         "detail": f"English inside the Luganda reply: {', '.join(leaks[:6])}"})
    elif leaks:
        problems.append({"level": "warn", "check": "english-leak",
                         "detail": f"possible English: {', '.join(leaks[:6])}"})

    if CLOCK_24.search(reply) or AM_PM.search(reply):
        problems.append({"level": "fail", "check": "clock-format",
                         "detail": "clock written in 24h/AM-PM form; must be ssaawa 6-to-6"})

    for needle in probe.get("must_include", []):
        if needle.lower() not in low:
            problems.append({"level": "fail", "check": "must-include",
                             "detail": f"missing {needle!r}"})
    for needle in probe.get("must_not_include", []):
        if needle.lower() in low:
            problems.append({"level": "fail", "check": "must-not-include",
                             "detail": f"contains forbidden {needle!r}"})

    max_words = probe.get("max_words")
    if max_words and words > max_words * 1.5:
        problems.append({"level": "fail", "check": "too-long",
                         "detail": f"{words} words for a {max_words}-word budget"})
    elif max_words and words > max_words:
        problems.append({"level": "warn", "check": "long",
                         "detail": f"{words} words for a {max_words}-word budget"})

    # certainty about unknowables (prices, weather) without a hedge
    if any(k in low for k in UNKNOWABLE) and any(c in low for c in CERTAINTY):
        if not any(h in low for h in ("siiknowo", "siyinza", "simanyi", "tekinakanika",
                                      "maybe", "kwa kya kyenkana")):
            problems.append({"level": "warn", "check": "false-certainty",
                             "detail": "sounds certain about something it cannot know"})
    return problems


def run(url: str, probes: list[dict], *, timeout: float = 120.0) -> list[dict]:
    import httpx

    results = []
    with httpx.Client(base_url=url.rstrip("/"), timeout=timeout) as client:
        for probe in probes:
            resp = client.post("/api/chat/sync", json={"message": probe["prompt"]})
            resp.raise_for_status()
            body = resp.json()
            reply = body.get("reply") or body.get("message") or ""
            results.append({
                "id": probe["id"],
                "prompt": probe["prompt"],
                "reply": reply,
                "tests": probe.get("tests", ""),
                "problems": check_reply(probe, reply),
            })
    return results


def summarize(results: list[dict]) -> dict:
    """One bucket per probe: a reply that fails is not also counted as merely warned."""
    levels = []
    for r in results:
        kinds = {p["level"] for p in r["problems"]}
        levels.append("fail" if "fail" in kinds else "warn" if "warn" in kinds else "clean")
    return {
        "probes": len(results),
        "failed": levels.count("fail"),
        "warned": levels.count("warn"),
        "clean": levels.count("clean"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000", help="running app")
    ap.add_argument("--probes", default=str(PROBES))
    ap.add_argument("--only", nargs="*", help="probe ids, e.g. c05 c27")
    ap.add_argument("--json", help="write full results here")
    ap.add_argument("--dry-run", action="store_true", help="list the probes, call nothing")
    args = ap.parse_args()

    probes = load_probes(Path(args.probes))
    if args.only:
        probes = [p for p in probes if p["id"] in set(args.only)]
    if not probes:
        print("No probes matched.")
        return 1
    if args.dry_run:
        for p in probes:
            flag = "" if p.get("verified") else "  [needs native correction]"
            print(f"{p['id']}  {p['prompt']}\n     tests: {p.get('tests', '')}{flag}")
        print(f"\n{len(probes)} probe(s).")
        return 0

    results = run(args.url, probes)
    summary = summarize(results)
    for r in results:
        mark = "FAIL" if any(p["level"] == "fail" for p in r["problems"]) else \
               ("warn" if r["problems"] else " ok ")
        print(f"\n[{mark}] {r['id']}  {r['prompt']}")
        print(f"        reply: {r['reply'][:160]}{'…' if len(r['reply']) > 160 else ''}")
        for p in r["problems"]:
            print(f"        {p['level']:4} {p['check']}: {p['detail']}")

    print(f"\n{summary['clean']} clean / {summary['warned']} warn / {summary['failed']} fail "
          f"of {summary['probes']} probes")
    print("This is a smoke test, not a verdict: native raters with VOICE-PLAN.md's rubric decide "
          "whether the Luganda is actually good.")
    if args.json:
        Path(args.json).write_text(json.dumps({"summary": summary, "results": results},
                                              ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"results -> {args.json}")
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
