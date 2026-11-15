# Brief: continue this project — training on the Luganda Bible

You are picking up a working repo, not a sketch. Everything below is either verified by a test
or was measured by hand; where it was measured, the number is quoted. Read this file, then
`AGENTS.md` (the standing rules), then only the plan file for the step you are on.

**Goal of this handoff:** take the Luganda Bible from *files a human will supply* to
*training data that has been validated, mixed and evaluated*, without breaking the rules in
`AGENTS.md`. Speech, the web UI and the corpus sources beyond the Bible are out of scope here.

---

## 0. State of the repo

```
python3 -m pytest tests -q          # must be green before and after every change
```

* 112 tests, offline, no API keys, ~3 s. If any test needs the network, it is a bug.
* `git log --oneline | head -1` — the last commit on this line of work. Keep the tree clean;
  commit in small steps with a sentence explaining *why*, not just what.
* Live app: `make run` → http://localhost:8000 (`GET /api/health` for counts).
* Pipeline that already exists and works (do not rewrite it, extend it):

| Area | Where | What it does |
|---|---|---|
| Bible ingest | `corpus/ingest_bible.py` | verse-numbered `.txt` → paragraphs, `kind: bible`, `register: biblical`, `verified: true`, licence + edition per record |
| Review loop | `corpus/make_review_sheet.py` | `verified` is flipped only by a human verdict file; codes are hash-of-id so approvals cannot drift |
| Corpus build | `corpus/build_corpus.py` | dedupe, boilerplate strip, retrieval docs, CPT text, **`--max-register-share biblical=0.25`** |
| Pairs → SFT | `finetune/ingest_bible_pairs.py` | verse-aligned EN↔LG → chat records, capped at 20% of the existing SFT set |
| Hygiene | `finetune/clean_dataset.py` | `ŋ`→`ng`, `ɲ`→`ny`, curly quotes, zero-width, drops unusable turns, backs up to `*.bak` |
| Validation | `finetune/validate_dataset.py` | SFT / eval / DPO schemas, duplicates, unverified fraction |
| Training | `finetune/train_lora.py`, `train_openai.py`, `configs/*.yaml` | QLoRA (GPU) or hosted fine-tune (no GPU) |
| Evaluation | `finetune/eval.py`, `finetune/probe_check.py` | chrF/BLEU + LLM judge, `human_rubric.csv`; 30 conversation probes with machine checks |
| Ground truth for "good" | `VOICE-PLAN.md` §Phase 1 | 5-axis native rubric, pass bar: mean ≥ 4.0, no axis < 3.5 |

**Do not re-litigate** the architecture: text-quality-first, retrieval before fine-tuning,
no GPU assumptions, handoff-friendly. The reasoning is written down in `TRAINING-OPTIONS.md`,
`CORPUS-PLAN.md` and `VOICE-PLAN.md`. If you think a decision is wrong, write the counter-argument
in a commit message or an issue, don't silently invert it.

---

## 1. Step 0 — the human's job, do not do it for them

The Bible files are **not** in the repo and must not be fetched by you. Two things only the
owner can decide:

1. **Which edition** — this is a licence fork, not a taste fork:

   | Edition | Status | Consequence |
   |---|---|---|
   | 1968 UBS / `lug` on ebible.org | public domain | no obligations; oldest spelling |
   | Biblica "Bayibuli Entukuvu" 1984–2014 | CC BY-SA 4.0 | attribution + share-alike; a *distributed dataset* inherits it |
   | Bible Society of Uganda 2003 | all rights reserved | needs written permission |

   If the answer is "I don't know", assume public domain 1968 and record that assumption in
   `corpus/README.md` next to the numbers.
2. **Where the files are** — expected layout, one file per book:

   ```
   bayibuli/01-Olwokusooka.txt
   bayibuli/02-Okuvaamu.txt
   …
   ```
   with each line starting `1:1 Olw'okusooka Katonda yatonda eggulu n'ensi.` (book name
   optional — the filename is used when it is absent). Wrapped continuation lines are joined
   automatically; page numbers and running heads are dropped.

**If the real files do not look like this, fix the parser, not the files.** Adapt
`VERSE_RE` / `book_name()` / `parse_verses()` in `corpus/ingest_bible.py` and add a test with a
trimmed real sample. Never hand-edit scripture to fit a regex, and never `sed` a corpus in
place — the ingest step is the only place text gets transformed, and it keeps locators so a
reviewer can always find the verse again.

---

## 2. Step 1 — ingest and prove what you got

```bash
python3 -m corpus.ingest_bible bayibuli/ --out bible --license "<decided above>"
python3 -m corpus.ingest_bible bayibuli/ --parallel english/          # only if a paired EN edition exists
```

**Acceptance, with numbers rather than adjectives:**

* verses parsed ≥ 30,000 (whole Bible ≈ 31,100); unparsed lines < 0.5% of all lines
* every record: `kind: bible`, `register: biblical`, `verified: true`, `license` and `edition` set
* words ≥ 1,300,000 (the text is ~1.6M words; below 1.3M something is being eaten)
* spot-check 3 paragraphs by eye against the source, including one with quotes and one across
  a chapter boundary
* `python3 -m corpus.ingest_bible bayibuli/ --dry-run` prints per-book verse counts — compare
  a few against a printed Bible

If a book parses to 0 verses, the script names it and continues. That is deliberate: a partial
ingest reported honestly beats a silent half-empty corpus.

---

## 3. Step 2 — build the corpus, with the register cap ON

```bash
python3 -m corpus.build_corpus --max-register-share biblical=0.25    # the default; leave it on
```

Why the cap exists, in one line: the Bible is 1.6M words of *attested but biblical* Luganda,
and fed in unbounded it becomes the only voice the model has — it starts answering "oli otya?"
like a pulpit. `cap_registers()` allows a register at most `cap/(1-cap) × words_of_everything_else`,
samples deterministically, and reports what it dropped.

**Acceptance:**

* `corpus/out/report.json` → `register_caps.biblical` shows `kept_words` and a `cap` of 0.25
* biblical share of `cpt/train.txt` ≤ 30% (whole segments can overshoot by one)
* if `corpus/out/report.json` says *"this register is the entire corpus"*, **stop** and tell the
  owner: with only scripture, the cap is skipped by design and the resulting voice will be
  one-note. The fix is more non-biblical text, not a different cap.
* retrieval (`retrieval/docs.jsonl`) is **not** capped — retrieval can hold everything; only
  training text is capped. Do not "fix" that.

Rough sizing, so you can answer "is this enough?" without guessing:

| Verified words | What it can support |
|---|---|
| < 50k | retrieval only; a fine-tune would memorise nothing useful |
| 50k – 1M | retrieval + SFT for style |
| > 1M | domain-adaptive pre-training becomes legitimate — and expensive (see Step 4) |

---

## 4. Step 3 — optional translation SFT, capped

Only if Step 1 produced parallel pairs.

```bash
python3 -m finetune.ingest_bible_pairs --dry-run     # shows the cap reasoning
python3 -m finetune.ingest_bible_pairs
```

Default cap: translation may add at most 20% of the *existing* SFT record count (measured from
disk, printed). A translator that answers "oli otya?" with a translation is the failure mode;
that is what the cap prevents. Vary the instruction templates, not the target text.

---

## 5. Step 4 — clean, validate, then decide whether to train at all

```bash
python3 -m finetune.clean_dataset finetune/data
python3 finetune/validate_dataset.py finetune/data        # gate: 0 errors
python3 finetune/build_dataset.py --verified-only
```

Known state before you start (measured, not assumed):

* `finetune/data/ingested/allandclive__alpaca_4k_luganda.jsonl` — 1169 records, 100% unverified,
  and it contained 12 records with `ŋ` (eng), a letter Luganda does not use. Already cleaned;
  keep it that way.
* `finetune/data/seed_luganda.jsonl` — 29 hand-written records, 100% unverified. **Unverified
  is not the same as bad** — it means no native speaker has signed it yet.
* `finetune/data/exported/chats_dpo.jsonl` — DPO schema; the validator now checks it on its own
  terms instead of counting 5 phantom errors.

**Then write the one-paragraph argument for your training decision before spending money.**
The honest default, given ~1.6M words and a laptop: **retrieval + prompt + few-shots already
beat a fine-tune**, and a fine-tune only earns its keep if it beats the prompt baseline on the
probe set (Step 5). Continued pre-training on 1.6M words is ~2.6M tokens — that is a rounding
error against a 14B model's pretraining, so it can shift style and register, and it cannot
teach the language. Say so in the commit message rather than overselling the run.

Routes, cheapest first (details and licences in `TRAINING-OPTIONS.md`):

1. hosted fine-tune (OpenAI mini-class / Together / Fireworks / Predibase) — `finetune/train_openai.py`,
   `make cloud-plan` validates the file with no network
2. free Colab/Kaggle T4 — `finetune/colab_qlora.ipynb`, LoRA on a small model
3. rented GPU by the hour — `finetune/configs/lora_*.yaml`; a T4 **cannot** hold Sunflower-14B
   (`lora_t4_8b.yaml` is the fallback, and that limit is measured, not guessed)

---

## 6. Step 5 — evaluation is the deliverable, not the checkpoint

```bash
python3 -m finetune.probe_check --url http://localhost:8000        # machine-checkable half
python3 finetune/eval.py --judge                                    # chrF/BLEU + LLM judge
# then: three native speakers fill finetune/out/human_rubric.csv — nothing replaces this
```

Report, in the commit or a `finetune/out/REPORT.md`: probes clean/warn/fail, chrF before and
after, and the rubric means per axis against the prompt-only baseline. **If the fine-tune does
not beat the baseline, keep the baseline** and say so plainly — that is a successful experiment,
not a failed one.

---

## 7. Things that will bite you (all observed here)

* FTS5 has no stemming — the app's search uses per-token terms plus a `LEXICON`; don't "simplify" it.
* The language ID accepts Runyankole as Luganda (score ~0.23). Documented, not a bug to chase.
* `lug.wikipedia.org` does not resolve in some sandboxes; test it on the user's machine.
* bukedde/newvision robots.txt disallows most article paths and both sitemaps return 500.
  Never add a bypass flag.
* No `ffmpeg`, no `espeak-ng`, no `python-multipart`, no `openai` SDK in this environment.
  HTTP is raw `httpx` by design; audio endpoints take JSON, not uploads.
* Tests must stay offline. Provider behaviour is tested through the mock or a local fake SSE server.
* One real bug from earlier: `FEW_SHOTS` in `app/prompts.py` were defined but never sent to the
  provider. If you touch prompts, assert on the outgoing request, not just the constant.

## 8. Do not do, no matter how reasonable it looks

1. Set `verified: true` on anything a human has not signed (the Bible is the sole exception,
   because a published translation is already attested — and it stays tagged `biblical`).
2. Train on text you have not validated (`validate_dataset.py` must say 0 errors).
3. Remove the register cap to "use all the data".
4. Feed ASR/spoken text into `cpt/train.txt` while it is unverified.
5. Ship a synthetic voice without the speaker's written consent, or scrape one from TV/TikTok/radio.
6. Present a fine-tune as better than the prompt baseline without probe evidence.
7. Add a network call to `tests/`.

## 9. Open questions to take back to the owner (do not assume an answer)

1. Which Bible edition — public domain 1968, or CC BY-SA Biblica (share-alike flows to any
   dataset you hand out)?
2. Who are the three native reviewers, and when can they do ~1 hour each on
   `corpus/out/review.html`?
3. GPU budget: zero (hosted fine-tune or Colab only), or a few rented hours?
4. Is the 1968 spelling acceptable in the product, or must it match school-register `ss`/`nn`?

## 10. Definition of done for this handoff

- [ ] `pytest` green, tree clean, commits explain decisions
- [ ] `corpus/out/report.json` shows a Bible ingest with verse counts, licence, edition, register cap
- [ ] `validate_dataset.py finetune/data` → 0 errors
- [ ] `build_dataset.py` output reproduces from scratch via `make` targets
- [ ] one training route actually run **or** a written, costed argument for not running one yet
- [ ] probe + rubric numbers recorded against the prompt-only baseline
- [ ] `corpus/README.md` / `finetune/README.md` updated with what was done and what it cost
