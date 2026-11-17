#!/usr/bin/env python3
"""Build the recording script for capturing a native speaker's voice.

    python3 -m voice.record_script                      # -> voice/out/recording-script.html
    python3 -m voice.record_script --lines 80 --out x.html
    python3 -m voice.record_script --from-verified-only # only your signed-off text

    # pull lines from specific corpus sources, balanced across them, Bible capped:
    python3 -m voice.record_script --from-verified-only --kinds bible,books,news --lines 200

Why a script at all: a voice clone is only as good as its coverage. Fifty sentences of
greetings teaches a model to greet and nothing else, and every number, every question
intonation and every geminate (``ss``, ``nn``) it never heard will come out wrong. This
picks lines that span short/long, statement/question and the sounds Luganda leans on, then
gives the reader something they can get through in one sitting.

Lines can come straight from the verified corpus (Bible, textbooks, newspapers) so every
recorded sentence is text a native speaker has already signed off — and every line in the
script carries its provenance (source kind and locator, e.g. the verse range or article)
into the manifest, so a mispronounced take can be traced back to where the text came from.

Two balance rules are enforced, because a raw corpus would break both:

1. **Per-source quota.** The Bible has a hundred times more lines than the textbooks;
   an unbalanced pick would teach the voice to read Scripture and nothing else.
2. **Biblical-register cap** (default a quarter of the lines, per CORPUS-PLAN.md) — a voice
   that only ever heard a pulpit will answer your users like one.

The output is deliberately boring: numbered lines, an estimate of how long each takes, and a
manifest so the audio can be aligned and checked afterwards. One file per line —
``0001.wav``, ``0002.wav`` … — is what every TTS trainer wants anyway.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

from corpus.make_review_sheet import load_segments

OUT_DIR = Path(__file__).resolve().parent / "out"
SCRIPT_HTML = OUT_DIR / "recording-script.html"
MANIFEST = OUT_DIR / "manifest.jsonl"

# Frames worth having even if the corpus does not contain them: they carry the sounds and
# the intonation patterns a chat assistant actually needs.
FRAMES = [
    "Oli otya? Nze Yogera, nsobola okukuyamba mu Luganda.",
    "Weebale nnyo. Kati ka tugende mu maaso.",
    "Ssaawa mmeka kati?", "Ssaawa emu ey'oku makya.", "Ssaawa kkumi n'emu ez'olweggulo.",
    "Ennaku z'omwezi ze ziti?",
    "Emmere eno ya ssente mmeka?", "Kikumi n'amakumi abiri.",
    "Nkusaba onnyambe, njagala okumanya ekifo kino.",
    "Tewali kizibu. Ka tukomewo ku kintu ekirala.",
    "Amazzi agakulukuta mu mugga Kiyira maanyi nnyo.",
    "Essanyu lyange lisinga okuba ku ssembya.",
    "Nnyabo, musaale, okuva wa?",
    "Abavubuka bangi bazannya omupiira ku lw'ezooba.",
    "Sitwala ssente nnyingi ku by'okulya wamu n'eby'okunywa.",
    "Kino kye kiseera ekirungi okusoma ebitabo byaffe.",
    "Yee, kituufu. Nedda, si kituufu.",
]


def sentences_from_corpus(verified_only: bool) -> list[str]:
    records, _ = load_segments()
    picked: list[str] = []
    for rec in records:
        if verified_only and not rec.get("verified"):
            continue
        for chunk in re.split(r"(?<=[.!?])\s+|\n+", rec.get("text", "")):
            chunk = chunk.strip()
            words = len(chunk.split())
            if 4 <= words <= 30:
                picked.append(chunk)
    return picked


def corpus_lines_tracked(*, verified_only: bool = True,
                          kinds: set[str] | None = None) -> list[dict]:
    """Sentence-level corpus lines, each carrying where it came from.

    Same splitting rule as ``sentences_from_corpus`` (4-30 words) — no new Luganda
    judgement happens here, only bookkeeping, so any line the reader stumbles on can be
    traced back to its verse range, textbook segment or article.
    """
    records, _ = load_segments()
    picked: list[dict] = []
    seen: set[str] = set()
    for rec in records:
        if verified_only and not rec.get("verified"):
            continue
        kind = rec.get("kind") or "unknown"
        if kinds and kind not in kinds:
            continue
        register = rec.get("register") or "plain"
        for chunk in re.split(r"(?<=[.!?])\s+|\n+", rec.get("text", "")):
            chunk = chunk.strip()
            words = len(chunk.split())
            if not (4 <= words <= 30) or chunk in seen:
                continue
            seen.add(chunk)
            picked.append({
                "text": chunk, "words": words,
                "source": rec.get("source", "unknown"), "kind": kind,
                "locator": rec.get("locator", ""), "register": register,
            })
    return picked


def select(lines: list[str], limit: int) -> list[str]:
    """Spread the pick across lengths so the voice hears short and long utterances."""
    by_len = sorted(set(lines), key=lambda s: (len(s.split()), s))
    if len(by_len) <= limit:
        return by_len
    step = len(by_len) / limit
    picked = [by_len[int(i * step)] for i in range(limit)]
    # keep first-seen order rather than alphabetical — reading order should feel natural
    order = {line: i for i, line in enumerate(lines)}
    return sorted(set(picked), key=lambda s: order.get(s, 10**9))


def select_tracked(items: list[dict], limit: int, *, max_biblical: float = 0.25) -> list[dict]:
    """Balanced pick across source kinds, with a hard cap on biblical-register lines.

    Proportional quotas per kind (at least one line each if the kind is present at all),
    length spread within each kind as ``select`` does, and the biblical share capped so a
    hundred Bible paragraphs cannot crowd out the textbooks and news the voice also needs.
    """
    if not items or limit <= 0:
        return []
    by_kind: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        by_kind[it.get("kind") or "plain"].append(it)

    quotas = {k: max(1, round(limit * len(v) / len(items))) for k, v in by_kind.items()}
    overflow = sum(quotas.values()) - limit
    if overflow > 0:                       # shrink the biggest pools first, keep one line each
        for k in sorted(quotas, key=lambda k: -len(by_kind[k])):
            if overflow <= 0:
                break
            cut = min(overflow, quotas[k] - 1)
            quotas[k] -= cut
            overflow -= cut

    picked: list[dict] = []
    for kind, lines in by_kind.items():
        ordered = sorted(lines, key=lambda i: (i["words"], i["text"]))
        step = max(1, math.ceil(len(ordered) / quotas[kind]))
        picked.extend(ordered[::step][:quotas[kind]])

    cap = int(limit * max_biblical)
    biblical = [p for p in picked if p.get("register") == "biblical"]
    if len(biblical) > cap:
        rest = [p for p in picked if p.get("register") != "biblical"]
        picked = rest[: max(0, limit - cap)] + biblical[:cap]

    picked = picked[:limit]
    order = {id(it): i for i, it in enumerate(items)}
    return sorted(picked, key=lambda it: order.get(id(it), 10**9))


def _provenance(m: dict) -> str:
    bits: list[str] = []
    if m.get("kind") and m["kind"] not in {"", "plain"}:
        bits.append(m["kind"])
    if m.get("locator"):
        bits.append(m["locator"])
    if m.get("source") and m["source"] not in {"", "frames", "plain"}:
        bits.append(m["source"])
    return " · ".join(bits)


def build(lines: list, *, minutes: float | None) -> tuple[str, list[dict]]:
    """lines may be plain strings (frames, old callers, tests) or tracked dicts."""
    norm: list[dict] = []
    for line in lines:
        if isinstance(line, dict):
            norm.append(line)
        else:
            norm.append({"text": line, "words": len(line.split()),
                        "source": "", "kind": "plain", "locator": ""})

    manifest: list[dict] = []
    total = 0.0
    for i, item in enumerate(norm, start=1):
        words = item.get("words") or len(item["text"].split())
        seconds = max(1.5, 0.42 * words + 0.8)
        total += seconds
        manifest.append({
            "id": f"{i:04d}", "file": f"{i:04d}.wav", "text": item["text"],
            "words": words, "est_seconds": round(seconds, 1),
            "source": item.get("source", ""), "kind": item.get("kind", ""),
            "locator": item.get("locator", ""),
        })

    rows = []
    for item in manifest:
        prov = _provenance(item)
        extra = f" · {html.escape(prov)}" if prov else ""
        rows.append(
            f"<li><div class=\"meta\"><span class=\"n\">{item['id']}</span>"
            f"<span class=\"t\">≈{item['est_seconds']:.0f}s · {item['words']} words{extra}</span></div>"
            f"<div class=\"line\">{html.escape(item['text'])}</div></li>"
        )

    kinds_used = sorted({m["kind"] for m in manifest if m["kind"] and m["kind"] != "plain"})
    origin = ", ".join(kinds_used) if kinds_used else "the fixed frames"

    page = f"""<!doctype html><html lang="lg"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Yogera — recording script</title><style>
:root {{ color-scheme: light; }}
body {{ margin: 0; padding: 34px 22px 80px; background: #f4f1ea; color: #1f1c17;
        font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Georgia, serif; }}
.wrap {{ max-width: 760px; margin: 0 auto; }}
h1 {{ font-size: 26px; margin: 0 0 4px; letter-spacing: -0.3px; }}
.sub {{ color: #6b6355; font-size: 15px; margin: 0 0 20px; }}
.box {{ background: #fffdf8; border: 1px solid #e0d8c8; border-radius: 12px; padding: 16px 20px;
        margin: 0 0 22px; font-size: 15px; }}
.box h2 {{ font-size: 15px; margin: 0 0 8px; text-transform: uppercase; letter-spacing: .08em;
           color: #8a7a58; }}
ol {{ list-style: none; padding: 0; margin: 0; counter-reset: none; }}
li {{ background: #fffdf8; border: 1px solid #e0d8c8; border-radius: 10px;
      padding: 12px 16px; margin: 0 0 10px; }}
.meta {{ display: flex; gap: 10px; align-items: center; margin-bottom: 4px; }}
.n {{ font: 700 12px/1 ui-monospace, Menlo, monospace; background: #1f1c17; color: #f4f1ea;
      padding: 3px 6px; border-radius: 4px; }}
.t {{ font-size: 12px; color: #8a8272; }}
.line {{ font-size: 19px; }}
.foot {{ color: #6b6355; font-size: 14px; }}
@media print {{ body {{ background: #fff; padding: 0; }} li {{ break-inside: avoid; }}
                .box {{ break-inside: avoid; }} }}
</style></head><body><div class="wrap">
<h1>Recording script</h1>
<p class="sub">{len(manifest)} lines · about {total/60:.0f} minutes of speech ·
~{total/60*1.6:.0f} minutes at the mic with pauses</p>

<div class="box"><h2>How to record</h2>
<ol style="list-style:decimal;padding-left:20px">
<li>Quiet room, soft furnishings, phone or laptop mic 20&nbsp;cm away. No music, no fan.</li>
<li>One file per line: <code>0001.wav</code>, <code>0002.wav</code> … (WAV or the highest-quality
    format your recorder offers — <b>not</b> MP3 at 64 kbps).</li>
<li>About half a second of silence before you start speaking and after you finish.</li>
<li>Read it like you are telling a friend, not announcing news. Natural pace beats perfect diction.</li>
<li>If you stumble: stop, breathe, read the line again from the beginning, keep going. Do not
    re-record the whole session.</li>
<li>Two takes of the first ten lines is a useful warm-up — the first ten are always stiff.</li>
</ol></div>

<div class="box"><h2>Before you start — consent</h2>
A cloned voice is a permanent, portable copy of somebody's identity. Record your own voice and
this is simple. Record a friend, a teacher or a relative and you need their <b>written</b> yes,
covering: what it will be used for, that it can be deleted on request, and that it will not be
used to make them say things they did not say. Never train a voice from TV, TikTok, radio or
church recordings — those people never agreed to it, and it is the one mistake in this whole
project that cannot be undone.
</div>

<div class="box"><h2>What happens next</h2>
The audio plus <code>voice/out/manifest.jsonl</code> (line text, duration, file name — and, for
corpus lines, where the text came from) is what a TTS fine-tune needs. Check the take with
<code>python3 -m voice.check_takes voice/out/</code> before spending any GPU time: a
mispronounced line teaches the model the mistake.</div>

<ol>{''.join(rows)}</ol>
<p class="foot">Generated from {html.escape(origin)} —
correct any line you would not say out loud before recording it.</p>
</div></body></html>"""
    return page, manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lines", type=int, default=120, help="how many lines to put in the script")
    ap.add_argument("--out", default=str(SCRIPT_HTML))
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--frames-only", action="store_true", help="skip corpus text, use the fixed frames")
    ap.add_argument("--from-verified-only", action="store_true",
                    help="only use text a native speaker has already signed off")
    ap.add_argument("--kinds", default=None, metavar="KINDS",
                    help="comma-separated corpus source kinds to use, e.g. bible,books,news (default: all)")
    ap.add_argument("--max-biblical", type=float, default=0.25, metavar="SHARE",
                    help="maximum share of biblical-register lines (default 0.25 — a pulpit voice "
                         "answers your users like a pulpit)")
    args = ap.parse_args()

    kinds = {k.strip() for k in args.kinds.split(",") if k.strip()} if args.kinds else None
    corpus_items = [] if args.frames_only else corpus_lines_tracked(
        verified_only=args.from_verified_only, kinds=kinds)
    frame_items = [{"text": t, "words": len(t.split()), "source": "frames", "kind": "frames",
                    "locator": "", "register": "assistant"} for t in FRAMES]

    missing = (kinds or set()) - {it["kind"] for it in corpus_items}
    for kind in sorted(missing):
        print(f"! no verified corpus lines of kind '{kind}' — check that you ingested it "
              f"and that a native speaker signed it off")

    lines = select_tracked(frame_items + corpus_items, args.lines, max_biblical=args.max_biblical)

    page, manifest = build(lines, minutes=None)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    man = Path(args.manifest)
    man.parent.mkdir(parents=True, exist_ok=True)
    with man.open("w", encoding="utf-8") as fh:
        for item in manifest:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    total = sum(i["est_seconds"] for i in manifest)
    per_kind: dict[str, int] = defaultdict(int)
    for m in manifest:
        if m["kind"]:
            per_kind[m["kind"]] += 1
    breakdown = ", ".join(f"{k}: {v}" for k, v in sorted(per_kind.items())) or "frames only"
    print(f"{len(manifest)} lines, ~{total/60:.0f} min of speech -> {out}")
    print(f"manifest -> {man}")
    print(f"sources: {breakdown}")
    if not corpus_items:
        print("! No corpus text used — this script only covers the fixed frames.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
