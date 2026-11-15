# Teaching it from textbooks, newspapers, TV and TikTok

What you described — feeding it books, papers, handbooks, shows, TikToks and web pages — is
not one job. It is a **corpus pipeline**, and that corpus feeds two very different things:

| | Grounding (retrieval) | Learning (training) |
|---|---|---|
| Weights change | no | yes |
| Works today | yes, minutes | no (hours–weeks + GPUs) |
| Good for | facts, prices, policies, current affairs | style, register, grammar, vocabulary |
| Can it go stale | no — re-index when the facts change | yes — a model trained on today's prices lies next month |
| Cost | $0 | $3–6 (one rented GPU run) → $1000s (pre-training) |

**The rule that follows:** newspapers and handbooks go to *retrieval*; textbooks do both; TV
and TikTok teach *how people actually talk*, never facts or spelling. Get that split right and
each cheap step makes the next one better.

```
textbooks.pdf ─┐
handbooks.pdf ─┼→ corpus/ingest_files.py ─┐
newspapers/   ─┘   (PDF/DOCX/EPUB/HTML/SRT)│
web pages ────→ corpus/ingest_web.py ──────┼→ corpus/build_corpus.py
TV/TikTok ────→ corpus/ingest_media.py ────┘        │
   (ASR or subtitles)                              ├→ retrieval/docs.jsonl → the app's memory (instant)
                                                   ├→ cpt/train.txt        → continued pre-training (later)
                                                   └→ report.json          → what you actually have
```

Every stage runs on a laptop. The only heavy step is the last one.

---

## Source by source: what each one is really worth

### 📚 Textbooks — the best Luganda you can get
- **Gives you:** clean, edited, modern written Luganda with consistent orthography, full
  noun-class agreement, and *explanations* of the language. Nothing else in your list comes
  close for grammar quality.
- **Process:** `corpus/ingest_files.py` (PDF/DOCX/EPUB). Most older Uganda textbooks are scans
  with no text layer — the script counts and reports those as `needs_ocr` instead of silently
  dropping them. OCR recipe below.
- **Watch out:** school books are formal and slightly dated ("ssomero" language, not "TikTok
  language"). Don't let them be your only source or the bot will sound like a 1990s exam paper.
- **Rights:** in copyright (NCDC/Longhorn/Fountain etc.). Buying them and using them for
  research is one thing; publishing a model trained on them or serving their text verbatim is
  another. Ask, or use the openly licensed ones.

### 📰 Newspapers (Bukedde, New Vision Luganda, Orumuri) — the modern written register
- **Gives you:** today's vocabulary — how "budget", "loan", "vaccination", "election" are
  actually written in Luganda *this year*. This is what users expect the bot to understand.
- **Process:** `corpus/ingest_web.py --urls …` or `--sitemap …`.
  **It obeys robots.txt, and I checked Bukedde for you:** `https://www.bukedde.co.ug/robots.txt`
  currently disallows `/news`, `/articledetails`, `/archives`, `/search` — i.e. the articles.
  So the polite paths are: email Bukedde/New Vision for archive or API access, buy a
  subscription, or scan the print edition. New Vision's robots file is permissive, but
  permissive robots is *not* permission to republish.
- **Best use:** retrieval first (news goes stale fast), plus style for SFT.
- **Watch out:** duplicated syndicated copy and repeated page furniture. The pipeline strips
  boilerplate (running heads, "© Bukedde", page numbers) and collapses near-duplicates
  automatically.

### 📗 Handbooks, guides, NGO/government booklets — facts worth grounding on
- **Gives you:** health, agriculture, civic information in plain Luganda — exactly the things
  people ask a chatbot. High value per word, and often published under open licences
  (ministries, NGOs, UNESCO).
- **Process:** same as textbooks. **Push straight into retrieval** — this is where a Luganda
  bot stops being a toy: "eddagala ly'omusujja ly'ani?" gets a real, sourced answer.
- **Rights:** check each one; many are freely reusable with attribution.

### 📺 TV shows (Bukedde TV, NTV Luganda news, drama, talk shows) — spoken register
- **Gives you:** natural turn-taking, greetings, jokes, argument, politeness levels, and the
  T/V distinction used correctly between an elder and a peer. Written corpora have almost none
  of this.
- **Process:** subtitles if you can get them (zero ASR error), otherwise ASR:
  `corpus/ingest_media.py --asr hf` uses **Sunbird's `asr-whisper-large-v3-salt`** (trained on
  Ugandan speech) or a local `faster-whisper` via `--asr-command`.
- **Watch out:** ASR mangles doubled consonants (`ssente` → `sente`), so **transcripts are
  never a spelling target**. The pipeline tags them `modality: spoken`, `verified: false`, and
  keeps them out of the pre-training text automatically.
- **Rights:** broadcast content. Recording for study is one thing; shipping a model trained on
  it is another. Ask the station — or record your own interviews (best option: pay 10 people
  to talk about their day for an hour).

### 📱 TikTok — how young Kampala actually talks
- **Gives you:** slang, English mixing, new loanwords, tone. This is where you learn that people
  say "boss" and "delivery" inside a Luganda sentence.
- **Process:** `corpus/ingest_media.py --url <tiktok url> --platform tiktok` (yt-dlp + ASR).
- **Watch out:** short clips, background music, shouting, spills — the worst signal-to-noise on
  your list. Keep it, but weight it for *style* only, and never for grammar or facts.
- **Rights:** someone else's video, often with licensed music. Use it to study; do not ship it.

### 🌍 "Stuff from Google" — mostly a trap, with exceptions
- **Reality:** the vast majority of pages mentioning Uganda are in English. Without filtering,
  you will build an English corpus with a Luganda name. `corpus/langid.py` scores every segment
  and rejects English, Swahili, and text with no Luganda signal — and reports *how much* it
  rejected, which is the number that tells you whether a scrape was worth it.
- **Worth it:** Luganda Wikipedia (CC BY-SA — legally the cleanest big source), government and
  NGO sites, church and cultural sites, online versions of the papers you have permission for.
- **Not worth it:** machine-translated pages (they read like English in Luganda words — this is
  exactly the flaw in the public "alpaca Luganda" datasets), SEO pages, and anything with the
  same paragraph repeated on ten domains.

---

## The commands (all run offline except the crawls)

```bash
# 1. books, handbooks, scanned papers, subtitle files
python -m corpus.ingest_files books/ handbooks/ newspapers/ tv/ --out files

# 2. permissioned web pages
python -m corpus.ingest_web --check-robots https://example.ug/news/123   # ask before crawling
python -m corpus.ingest_web --sitemap https://example.ug/sitemap.xml --limit 200 --out web

# 3. TV / TikTok (subtitles if you have them; otherwise ASR)
python -m corpus.ingest_media --transcript shows/ep1.srt --kind tv --out media
python -m corpus.ingest_media --url https://www.tiktok.com/@x/video/123 --platform tiktok --out media

# 4. merge, dedupe, get the honest report
python -m corpus.build_corpus
python -m corpus.build_corpus --push-to-app http://localhost:8000   # grounds the bot now
```

`corpus/out/report.json` tells you what you own: words by kind, code-mixed share, spoken share,
duplicates removed, and how many segments the language filter threw away.

## OCR for scanned textbooks (the step people forget)

1. Get 300dpi greyscale scans. Straighten pages; OCR hates skew.
2. Run one of:
   - **Google Cloud Vision / Azure Document Intelligence** — best Latin-script accuracy,
     handles Luganda, costs ~$1.50 per 1,000 pages.
   - **A vision LLM** (Gemini/GPT-4o class) page by page — good on layout, slower, pricier.
   - **Tesseract** with `-l eng` plus a Luganda wordlist for post-correction — free, but it will
     break geminates (`ss`, `nn`, `kk`) and apostrophes; expect 5–15% word errors.
3. **Then run the validator, not your eyes:** `python finetune/validate_dataset.py corpus/out`
   and sample 100 lines. If the geminates look wrong, the OCR is wrong.
4. Mark OCR'd text `verified: false` and keep it out of spelling-critical training until a human
   checks a sample.

## How much of this do you actually need?

| Goal | Rough size | Effort |
|---|---|---|
| Ground the bot on your own facts (prices, policies, handbook) | 20k–200k words | a weekend |
| Stop it sounding like a translation (style SFT on curated chats) | 2,000–10,000 verified turns | weeks of review |
| Add spoken/register range | 10–50 hours of transcribed speech | a month, plus reviewers |
| Make the *model itself* better at Luganda (continued pre-training) | hundreds of millions of tokens | GPUs, weeks, real money |

The fourth row is the one people assume they need first. It is the last one you should do —
`report.json` will tell you if you are anywhere near it (you almost certainly are not yet, and
you do not need to be to have a good bot).

## The order I would actually work in

1. **Handbooks + your own facts → retrieval.** One evening. The bot becomes useful.
2. **Newspapers (with permission) → retrieval + style.** Ongoing.
3. **Textbooks → OCR if needed → retrieval now, CPT later.**
4. **A few TV episodes → ASR → review a sample → SFT for register.**
5. **TikTok → ASR → style only, heavily filtered.**
6. **Only then:** fine-tune (QLoRA or hosted), evaluate, and compare against the prompt +
   retrieval baseline. If the fine-tune does not beat it, keep the baseline — it is cheaper and
   it updates instantly.

---

### One honest summary

Your instinct is right: real Luganda content is the only way to get a bot that sounds like
Kampala. But volume is not the goal — **verified** volume is. Ten thousand sentences a native
speaker has signed off is worth more than a million words of scraped, OCR'd, machine-translated
text, and it is the only thing that will not make the model confidently wrong in a language
where its mistakes are invisible to everyone but the people you are serving.
