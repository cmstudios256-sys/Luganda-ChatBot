# corpus/ — turning books, papers, shows and TikToks into usable Luganda

Full strategy and the per-source trade-offs: **`../CORPUS-PLAN.md`**. This file is the cheat sheet.

```bash
# books / handbooks / newspapers / subtitle files
python -m corpus.ingest_files books/ handbooks/ newspapers/ tv/ --out files

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
