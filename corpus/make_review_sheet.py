#!/usr/bin/env python3
"""Turn the corpus into a review sheet a native speaker can actually get through.

    python -m corpus.make_review_sheet                 # -> corpus/out/review.html
    python -m corpus.make_review_sheet --status        # how much is signed off
    python -m corpus.make_review_sheet --apply corpus/out/verdicts.txt

The point of this file is the word *verified*. Right now almost every record is
``verified: false``, which is honest but useless: nobody can train on it and nobody knows
which 200 lines to look at first. This closes the loop end to end:

    build segments -> read the sheet -> mark each item -> apply verdicts -> train

How the sheet works offline (which it must — the preview sandbox has no network):

* Every record gets a short code (``A3F2``). Codes are derived from the segment id, so a
  code still means the same thing tomorrow.
* Read it, tick **ok** / **fix** / **reject**. If you tick *fix*, type the corrected text
  in the box; what you type becomes the training text.
* Press **Build verdicts** and copy the text box. Save it as ``corpus/out/verdicts.txt``.
* ``--apply`` reads it back. Corrected and approved records come out ``verified: true``;
  rejected ones are dropped from training but kept on disk so nothing quietly vanishes.

One rule this script enforces rather than trusts: nothing is marked verified for you.
If you do not sign a line, it stays unverified and the trainer skips it.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .pipeline import OUT, SEGMENTS

REVIEW_HTML = OUT / "review.html"
SEGMENT_GLOB = "*.jsonl"


# --------------------------------------------------------------------------- #
# Loading / codes
# --------------------------------------------------------------------------- #
def load_segments(paths: list[Path] | None = None) -> tuple[list[dict], dict[str, Path]]:
    """Read every segment file. Returns (records, id -> file it lives in)."""
    files = paths or sorted(SEGMENTS.glob(SEGMENT_GLOB))
    records: list[dict] = []
    home: dict[str, Path] = {}
    for path in files:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                records.append(rec)
                home[rec["id"]] = path
    return records, home


def _code(rec_id: str, width: int) -> str:
    """A short code that depends on the record alone, never on its neighbours.

    This matters more than it looks: if codes were derived from position or from a
    record's first N characters, adding a file between printing the sheet and applying
    the verdicts could attach your approval to a different paragraph. A hash of the id
    cannot move when the corpus grows.
    """
    return hashlib.sha256(rec_id.encode("utf-8")).hexdigest()[:width].upper()


def assign_codes(records: list[dict]) -> dict[str, str]:
    """id -> short code, unique within this set; widens only where two ids collide."""
    width = 5
    while True:
        codes = {rec["id"]: _code(rec["id"], width) for rec in records}
        if len(set(codes.values())) == len(codes) or width >= 32:
            return codes
        width += 1


# --------------------------------------------------------------------------- #
# The sheet
# --------------------------------------------------------------------------- #
CARD_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; padding: 32px 24px 120px; background: #f4f1ea; color: #1f1c17;
       font: 16px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Georgia, serif; }
.wrap { max-width: 820px; margin: 0 auto; }
h1 { font-size: 27px; margin: 0 0 6px; letter-spacing: -0.3px; }
.sub { color: #6b6355; margin: 0 0 22px; font-size: 15px; }
.rule { border: 0; border-top: 2px solid #d9d2c4; margin: 24px 0; }
.card { background: #fffdf8; border: 1px solid #e0d8c8; border-radius: 12px;
        padding: 18px 20px 14px; margin: 0 0 16px; box-shadow: 0 1px 2px rgba(60,50,30,.05); }
.head { display: flex; flex-wrap: wrap; gap: 8px; align-items: baseline; margin-bottom: 10px; }
.code { font: 700 13px/1 ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: 1px;
        background: #1f1c17; color: #f4f1ea; padding: 4px 7px; border-radius: 5px; }
.tag { font-size: 12px; color: #6b6355; background: #efe9dc; border-radius: 20px; padding: 3px 9px; }
.tag.good { background: #dfeede; color: #2f5a34; }
.tag.warn { background: #f6e6cf; color: #7a5320; }
.text { white-space: pre-wrap; font-size: 17px; }
.note { font-size: 13px; color: #8a8272; margin-top: 8px; font-style: italic; }
.verdicts { display: flex; flex-wrap: wrap; gap: 14px; align-items: center; margin-top: 14px;
            padding-top: 12px; border-top: 1px dashed #ded6c6; font-size: 14px; }
.verdicts label { display: inline-flex; gap: 5px; align-items: center; cursor: pointer; }
.verdicts .ok b { color: #2f7a3a; } .verdicts .fix b { color: #9a6b12; } .verdicts .no b { color: #a3372f; }
textarea.fixbox { width: 100%; margin-top: 10px; min-height: 78px; padding: 9px; font: 14px/1.5 inherit;
                  border: 1px solid #d8d0c0; border-radius: 8px; background: #fffdf8; display: none; }
textarea.fixbox.on { display: block; }
.bar { position: fixed; left: 0; right: 0; bottom: 0; background: #1f1c17; color: #f4f1ea;
       padding: 12px 20px; display: flex; gap: 14px; align-items: center; flex-wrap: wrap;
       font-size: 14px; }
.bar button { font: 600 14px inherit; background: #e8c26a; color: #241f16; border: 0;
              padding: 9px 15px; border-radius: 8px; cursor: pointer; }
.bar button.ghost { background: transparent; color: #e8c26a; border: 1px solid #6b6355; }
.bar .count { margin-left: auto; color: #b9b1a0; }
#out { width: 100%; margin-top: 10px; min-height: 90px; display: none; padding: 10px;
       font: 13px/1.5 ui-monospace, Menlo, monospace; border-radius: 8px; border: 1px solid #6b6355;
       background: #2a251d; color: #f4f1ea; }
#out.on { display: block; }
.empty { padding: 40px; text-align: center; color: #6b6355; }
@media print { .bar, .verdicts { display: none; } body { padding: 0; background: #fff; }
               .card { break-inside: avoid; box-shadow: none; } }
"""

SHEET_JS = """
function wire(){
  var cards = [].slice.call(document.querySelectorAll('.card'));
  cards.forEach(function(c){
    var box = c.querySelector('.fixbox');
    var fixRadio = c.querySelector('input[value="fix"]');
    if(!fixRadio || !box) return;
    fixRadio.addEventListener('change', function(){ box.classList.add('on'); box.focus(); });
    c.querySelectorAll('input[type=radio]').forEach(function(r){
      if(r.value !== 'fix') r.addEventListener('change', function(){ box.classList.remove('on'); });
    });
  });
}
function markRestOk(){
  var cards = document.querySelectorAll('.card');
  var marked = 0;
  cards.forEach(function(c){
    var picked = c.querySelector('input[type=radio]:checked');
    if(!picked){
      var ok = c.querySelector('input[value="ok"]');
      if(ok){ ok.checked = true; marked++; }
    } else { marked++; }
  });
  document.getElementById('count').textContent = marked + ' of ' + cards.length + ' marked';
}
function collect(){
  var lines = [];
  document.querySelectorAll('.card').forEach(function(c){
    var code = c.dataset.code;
    var picked = c.querySelector('input[type=radio]:checked');
    if(!picked) return;
    if(picked.value === 'ok')  lines.push(code + ' ok');
    if(picked.value === 'no')  lines.push(code + ' no');
    if(picked.value === 'fix'){
      var fix = (c.querySelector('.fixbox').value || '').replace(/\\s*\\n\\s*/g, ' ').trim();
      lines.push(code + (fix ? ' fix: ' + fix : ' fix:'));
    }
  });
  var out = document.getElementById('out');
  out.value = lines.join('\\n');
  out.classList.add('on');
  out.focus(); out.select();
  document.getElementById('count').textContent = lines.length + ' of ' +
    document.querySelectorAll('.card').length + ' marked';
}
window.addEventListener('DOMContentLoaded', wire);
"""


def build_sheet(records: list[dict], codes: dict[str, str], *, limit: int | None = None,
                only_unverified: bool = True) -> str:
    rows = [r for r in records if (not only_unverified or not r.get("verified"))]
    rows.sort(key=lambda r: (r.get("kind", ""), r["id"]))
    if limit:
        rows = rows[:limit]

    by_kind = Counter(r.get("kind", "?") for r in rows)
    words = sum(r.get("words", 0) for r in rows)
    total_words = sum(r.get("words", 0) for r in records)
    verified_words = sum(r.get("words", 0) for r in records if r.get("verified"))

    out = [
        "<!doctype html><html lang=\"lg\"><head><meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        "<title>Yogera — review sheet</title><style>", CARD_CSS, "</style></head><body><div class=\"wrap\">",
        "<h1>Review what it learned</h1>",
        f"<p class=\"sub\">{len(rows)} item(s) to read · {words:,} words in this sheet · "
        f"{verified_words:,} of {total_words:,} words in the corpus are already signed off.</p>",
        "<p class=\"sub\">Tick <b>ok</b> if the Luganda is good, <b>fix</b> if it needs a "
        "correction (type the corrected text), <b>reject</b> if it should never be trained on. "
        "Then press <b>Build verdicts</b>, copy the box, and save it as "
        "<code>corpus/out/verdicts.txt</code>.</p>",
        "<hr class=\"rule\">",
    ]
    for kind, count in by_kind.most_common():
        out.append(f"<span class=\"tag\">{html.escape(str(kind))}: {count}</span> ")

    if not rows:
        out.append("<div class=\"empty\">Nothing left to review — everything in the corpus is "
                   "already signed off.</div>")

    for rec in rows:
        text = rec.get("text", "")
        kind = rec.get("kind", "?")
        lang = rec.get("lang") or {}
        score = lang.get("lug_score")
        tags = [f"<span class=\"tag\">{html.escape(str(rec.get('source', 'unknown')))}"
                f" · {html.escape(str(rec.get('locator', '')))}</span>",
                f"<span class=\"tag\">{rec.get('words', 0)} words</span>"]
        if score is not None:
            cls = "tag good" if score >= 0.15 else "tag warn"
            tags.append(f"<span class=\"{cls}\">Luganda score {score:.2f}</span>")
        if lang.get("code_mix"):
            tags.append("<span class=\"tag warn\">code-mixed</span>")
        if rec.get("register"):
            tags.append(f"<span class=\"tag\">{html.escape(str(rec['register']))}</span>")
        if rec.get("license"):
            tags.append(f"<span class=\"tag\">{html.escape(str(rec['license']))}</span>")

        code = codes.get(rec["id"], rec["id"][:4].upper())
        out += [
            f"<div class=\"card\" data-code=\"{code}\">",
            f"<div class=\"head\"><span class=\"code\">{code}</span>"
            f"<span class=\"tag\">{html.escape(str(kind))}</span>{''.join(tags)}</div>",
            f"<div class=\"text\">{html.escape(text)}</div>",
            "<textarea class=\"fixbox\" placeholder=\"Type the correct Luganda for this passage…\">"
            f"{html.escape(text)}</textarea>",
            "<div class=\"verdicts\">",
            "<label class=\"ok\"><input type=\"radio\" name=\"" + code + "\" value=\"ok\"> <b>ok</b></label>",
            "<label class=\"fix\"><input type=\"radio\" name=\"" + code + "\" value=\"fix\"> <b>fix</b></label>",
            "<label class=\"no\"><input type=\"radio\" name=\"" + code + "\" value=\"no\"> <b>reject</b></label>",
            "</div></div>",
        ]

    out += [
        "</div>",
        "<div class=\"bar\"><button onclick=\"collect()\">Build verdicts</button>",
        "<button class=\"ghost\" onclick=\"markRestOk()\">Mark unmarked ok</button>",
        "<button class=\"ghost\" onclick=\"window.print()\">Print</button>",
        "<textarea id=\"out\" spellcheck=\"false\"></textarea>",
        "<span class=\"count\" id=\"count\">nothing marked yet</span></div>",
        "<script>", SHEET_JS, "</script></body></html>",
    ]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Verdicts
# --------------------------------------------------------------------------- #
VERDICT_RE = re.compile(r"^\s*([0-9A-Za-z]+)\s*(ok|no|fix)\s*:?\s*(.*)$", re.IGNORECASE)


def parse_verdicts(text: str) -> dict[str, tuple[str, str]]:
    """'A3F2 ok' / 'A3F2 no' / 'A3F2 fix: corrected words' -> {code: (verdict, new_text)}"""
    out: dict[str, tuple[str, str]] = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        m = VERDICT_RE.match(line)
        if not m:
            print(f"  ? could not read verdict line: {line!r}")
            continue
        code, verdict, payload = m.group(1).upper(), m.group(2).lower(), m.group(3).strip()
        if verdict == "fix" and not payload:
            print(f"  ? {code}: 'fix' without corrected text — treating as rejected (nothing to train on)")
            verdict = "no"
        out[code] = (verdict, payload)
    return out


def apply_verdicts(verdict_text: str, *, files: list[Path] | None = None) -> dict:
    records, home = load_segments(files)
    codes = assign_codes(records)
    code_to_id = {v: k for k, v in codes.items()}
    verdicts = parse_verdicts(verdict_text)

    by_file: dict[Path, list[dict]] = defaultdict(list)
    for rec in records:
        by_file[home[rec["id"]]].append(rec)

    summary = Counter()
    unknown = [c for c in verdicts if c not in code_to_id]
    for code, (verdict, payload) in verdicts.items():
        rec_id = code_to_id.get(code)
        if rec_id is None:
            continue
        rec = next(r for r in records if r["id"] == rec_id)
        if verdict == "ok":
            rec["verified"] = True
            rec["review"] = "approved"
            summary["approved"] += 1
        elif verdict == "no":
            rec["verified"] = False
            rec["review"] = "rejected"
            summary["rejected"] += 1
        else:  # fix
            rec["text"] = payload
            rec["words"] = len(payload.split())
            rec["verified"] = True
            rec["review"] = "corrected"
            summary["corrected"] += 1
        summary["words_signed"] += rec.get("words", 0)

    touched = {home[next(r["id"] for r in records if r["id"] == code_to_id[c])]
               for c in verdicts if c in code_to_id}
    for path in touched & set(by_file):
        backup = path.with_suffix(path.suffix + ".bak")
        if not backup.exists():
            backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        with path.open("w", encoding="utf-8") as fh:
            for rec in by_file[path]:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        summary["files_rewritten"] += 1

    if unknown:
        print(f"! {len(unknown)} code(s) in the verdict file do not exist here: "
              f"{', '.join(sorted(unknown)[:8])}")
    return dict(summary)


def status(records: list[dict]) -> str:
    per_kind: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        per_kind[rec.get("kind", "?")].append(rec)
    lines = [f"{'kind':<14}{'items':>7}{'words':>10}{'verified words':>16}{'% signed':>10}",
             "-" * 57]
    tot = ver = 0
    for kind, recs in sorted(per_kind.items(), key=lambda kv: -sum(r.get("words", 0) for r in kv[1])):
        w = sum(r.get("words", 0) for r in recs)
        v = sum(r.get("words", 0) for r in recs if r.get("verified"))
        tot += w
        ver += v
        pct = f"{100 * v / w:.0f}%" if w else "-"
        lines.append(f"{kind:<14}{len(recs):>7}{w:>10,}{v:>16,}{pct:>10}")
    lines.append("-" * 57)
    lines.append(f"{'total':<14}{len(records):>7}{tot:>10,}{ver:>16,}"
                 f"{(f'{100 * ver / tot:.0f}%' if tot else '-'):>10}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--segments", nargs="*", help="segment .jsonl files (default: corpus/out/segments/*.jsonl)")
    ap.add_argument("--out", default=str(REVIEW_HTML), help="where to write the sheet")
    ap.add_argument("--limit", type=int, help="only the first N items (to taste the format)")
    ap.add_argument("--all", action="store_true", help="include items already signed off")
    ap.add_argument("--apply", metavar="VERDICTS_FILE", help="read verdicts back and update segments")
    ap.add_argument("--status", action="store_true", help="print verification coverage and exit")
    args = ap.parse_args()

    files = [Path(p) for p in args.segments] if args.segments else None
    records, _ = load_segments(files)
    if not records:
        print(f"No segments under {SEGMENTS}. Build some first: python -m corpus.build_corpus")
        return 1

    if args.apply:
        path = Path(args.apply)
        if not path.exists():
            print(f"No such verdict file: {path}")
            return 1
        summary = apply_verdicts(path.read_text(encoding="utf-8"), files=files)
        print("applied:", ", ".join(f"{k}={v}" for k, v in sorted(summary.items())) or "nothing")
        recs_after, _ = load_segments(files)
        print()
        print(status(recs_after))
        return 0

    if args.status:
        print(status(records))
        return 0

    codes = assign_codes(records)
    sheet = build_sheet(records, codes, limit=args.limit, only_unverified=not args.all)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(sheet, encoding="utf-8")
    pending = sum(1 for r in records if not r.get("verified"))
    print(f"Wrote {out} — {pending} item(s) still unsigned.")
    print("Read it, tick each item, press 'Build verdicts', save the box as "
          f"{OUT / 'verdicts.txt'}, then:")
    print(f"  python -m corpus.make_review_sheet --apply {OUT / 'verdicts.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
