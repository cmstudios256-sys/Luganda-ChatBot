# corpus/ — turning books, papers, shows and TikToks into usable Luganda

Full strategy and the per-source trade-offs: **`../CORPUS-PLAN.md`**. This file is the cheat sheet.

```bash
# books / handbooks / newspapers / subtitle files
python -m corpus.ingest_files books/ handbooks/ newspapers/ tv/ --out files

# the Luganda Bible: verse-numbered .txt per book, paragraphs rebuilt for you
python -m corpus.ingest_bible bayibuli/ --license "public domain (UBS 1968)"
python -m corpus.ingest_bible bayibuli/ --parallel english/   # + EN->LG verse pairs

# text you can personally vouch for (your own writing, a proofread book):
# sets verified:true so it can be used as a training target
python -m corpus.ingest_files corpus/samples --out samples --verified

# web (robots-aware; check first)
python -m corpus.ingest_web --check-robots https://example.ug/news/123
python -m corpus.ingest_web --sitemap https://example.ug/sitemap.xml --limit 200 --out web

# TV / TikTok: subtitles if you have them, ASR otherwise
python -m corpus.ingest_media --transcript shows/ep1.srt --kind tv --out media
python -m corpus.ingest_media --url https://www.tiktok.com/@x/video/123 --platform tiktok --dry-run
python -m corpus.ingest_media --file clip.m4a --asr hf            # Sunbird Luganda Whisper, needs HF_TOKEN
python -m corpus.ingest_media --file clip.m4a --asr-command \
    "faster-whisper --model large-v3 --language lg --output_format srt {out} {input}"

# merge → dedupe → report → (optionally) load into the running app
python -m corpus.build_corpus
python -m corpus.build_corpus --push-to-app http://localhost:8000
```

| File | What it does |
|---|---|
| `langid.py` | Luganda vs English vs Swahili vs code-mixed. Run it alone: `python corpus/langid.py` |
| `text_extract.py` | PDF, DOCX, EPUB, HTML, TXT, SRT/VTT, CSV, JSONL → segments with page/timecode |
| `ingest_files.py` | folder walker + language filter |
| `ingest_web.py` | robots.txt-obeying crawler (urllib.robotparser, crawl-delay, contact UA) |
| `ingest_media.py` | yt-dlp + ASR (Hugging Face or any local CLI) → timecoded transcripts |
| `pipeline.py` | normalization, language filtering, provenance, segment writer |
| `build_corpus.py` | boilerplate stripping, dedupe, retrieval docs, CPT text, report |

## Outputs (`corpus/out/`)

| Path | Use |
|---|---|
| `segments/*.jsonl` | raw kept segments from each ingest run |
| `corpus.jsonl` | everything, deduped, with provenance (`source`, `locator`, `lang`, `verified`) |
| `retrieval/docs.jsonl` | documents for `POST /api/knowledge/bulk` — grounds the bot immediately |
| `cpt/train.txt`, `cpt/val.txt` | plain text for continued pre-training (text-only; ASR excluded) |
| `report.json` | words by kind, code-mixed %, spoken %, duplicates, rejections |

## Rules baked into the code

1. **Every record is `verified: false`** until a native speaker signs it off.
2. **ASR transcripts never enter the pre-training text**, and are tagged `modality: spoken`.
3. **robots.txt is obeyed**, and disallowed URLs are counted, not fetched.
4. **The language filter reports its rejections.** A "huge" scrape that keeps 4% of its segments
   is a bad scrape — the report says so.
5. **Boilerplate is stripped, but content is never silently deleted** — if removing repeated
   lines would gut a segment, the original is kept and flagged instead.

## Signing text off

Nothing scraped or imported is trained on until a native speaker signs it. The Bible is the
exception (a published translation is already attested); it comes in `verified: true` and
tagged `register: biblical` so it stays a minority of any fine-tuning mix.

```bash
python -m corpus.make_review_sheet           # -> corpus/out/review.html (read, tick, copy)
python -m corpus.make_review_sheet --apply corpus/out/verdicts.txt
python -m corpus.make_review_sheet --status
```

Verdict lines: `CODE ok`, `CODE no`, `CODE fix: <corrected text>`. Segment files are backed
up to `*.jsonl.bak` before being rewritten.
