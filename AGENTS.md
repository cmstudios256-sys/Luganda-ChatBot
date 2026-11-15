# AGENTS.md — working notes for any coding agent on this repo

Read this before you change anything. Yogera is a Luganda-language product: the
hard part is not the code, it is **not inventing Luganda**. Most regressions here
are language regressions, and they pass every test unless you keep the rules below.

## What this is

- `app/` — FastAPI service: chat (SSE streaming), feedback, export of conversations
  to training data. SQLite, no ORM, no migrations. Entry point `app/main.py`.
- `app/prompts.py` — **the product.** A long, linguistically explicit Luganda system
  prompt: noun-class concords, tense/agreement grids, register (singular vs plural),
  greetings by time of day, orthography (geminates, apostrophes, no `ŋ`), refusal
  policy, few-shots. Treat edits here as changes to the language, not to a string.
- `app/providers.py` — OpenAI / Anthropic / Gemini / any OpenAI-compatible endpoint /
  offline `mock`. All behind one `stream()` async generator. httpx only, no vendor SDKs.
- `app/store.py` — SQLite persistence + the export that turns chats into
  `{messages:[...]}` SFT rows and `(prompt, chosen, rejected)` DPO pairs.
- `app/knowledge.py` — **the no-GPU learning path**: FTS5 document retrieval + exemplars
  (👍 answers and human rewrites) injected into the system prompt, plus a Luganda↔English
  glossary (`LEXICON`) that bridges retrieval across languages. Swapping the retriever for
  embeddings starts and ends in `search_documents()`.
- `web/index.html` — single file UI, inline CSS/JS, no build step, no CDN, no external
  fonts. Keep it that way (the preview iframe has no network access).
- `finetune/` — offline data + training pipeline. Its own README explains the sequence
  and the per-dataset licences. `--dry-run` works with no GPU; real training does not.

## Commands

```bash
pip install -r requirements.txt      # app deps (fastapi, uvicorn, httpx, pydantic)
pytest -q                            # 14 tests, offline, ~1s — run before every commit
./run.sh                             # serve on 0.0.0.0:8000
make validate                        # dataset checks (Luganda-aware)
make build                           # dedupe + split -> finetune/out/{train,val,test}.jsonl
make train-plan                      # QLoRA plan without downloading a model
python finetune/eval.py --predictions preds.jsonl   # chrF++ / BLEU, no key needed
python finetune/train_openai.py --dry-run           # hosted fine-tune: validates files, no network
```

Without any API key the app runs in **demo mode**: `MockProvider` returns canned
Luganda keyed on keywords. The UI and API label this `demo`. Never let demo output
leak into datasets or screenshots presented as real model output.

## Invariants — do not break these

1. **No invented Luganda.** If you add or edit Luganda text (prompt examples, seed
   data, mock replies), it must be text you can defend or clearly mark
   `"verified": false`. Do not "improve" grammar you are unsure about; simplify the
   sentence instead. Do not add proverbs unless a native speaker supplied them.
2. **Every dataset record carries `verified`.** `false` = awaiting a native reviewer.
   `build_dataset.py --verified-only` is how that flag gets enforced.
3. **Register must stay switchable.** `tone: auto|peer|respect` maps to the
   singular/plural (`-mu-`) split. Anything that hardcodes one register is a bug:
   addressing an elder with `Oli otya?` is rude, not informal.
4. **Answer in Luganda only**, with one English gloss sentence only when the user wrote
   English or asked for it. No unnecessary English words inside Luganda sentences.
5. **The loss is computed on assistant tokens only** (`build_response_collator` in
   `finetune/train_lora.py`). Removing that masking teaches instruction-echoing.
6. **Offline-first tests.** No test may require a network or an API key. Provider
   behaviour is tested through the mock or a local fake server.
7. **UI has no external dependencies.** Inline everything; no fonts, CDNs or images by URL.
8. **Memory is not training.** 👍/👎/corrections must keep feeding `knowledge.add_exemplar`,
   and every reply must still be generated from the same prompt+schema. Do not silently
   switch the product to "the fine-tune" — `TRAINING-OPTIONS.md` explains which route is
   active and why.

## Where to extend, and how

| Task | Do this | Do not |
|---|---|---|
| Add a language rule | Append to the relevant numbered section in `app/prompts.py`, add a `test_system_prompt_*` assertion | Rewrite the whole prompt and drop the grammar tables |
| Add a provider | Subclass `BaseProvider`, implement `stream()`, register in `get_provider()` + `KEY_ENV` in `config.py` | Add a vendor SDK; keep it httpx |
| Add a corpus | `finetune/ingest_hf.py --dataset ... --preset ...` then `validate_dataset.py`, then record the licence in `finetune/README.md` | Commit the downloaded corpus (gitignored, and licences vary) |
| Train without a GPU | `finetune/train_openai.py` (hosted API) or `finetune/colab_qlora.ipynb` (free T4) | Claim a T4 can train Sunflower-14B — it cannot fit |
| Change training | Edit `finetune/configs/*.yaml`; verify with `--dry-run` first | Bump the LR to "speed things up" — it washes out the base Luganda |
| Improve retrieval | `app/knowledge.py`: `LEXICON` for vocabulary gaps, `search_documents()` to change the ranker | Add a vector DB before trying keyword+glossary |
| Add voice | Whisper-Luganda in, VITS-Luganda out (`Sunbird/asr-whisper-large-v3-salt`, `Sunbird/tts-vits-lug`) | Invent a TTS pipeline before checking those exist |

## Definition of done

- `pytest -q` green.
- Any new Luganda string is either defensible or flagged unverified.
- If you touched the prompt or training data, run `python finetune/eval.py` against a
  fixed prediction file and report the chrF++ delta — not a vibe.
- If you touched the UI, screenshot it and check the empty state, a streamed reply, and
  the 👎 → correction flow.

## Facts worth having on hand

- Base models for Luganda: `Sunbird/Sunflower-14B` / `-32B` (Qwen3-based, 31 Ugandan
  languages, Apache-style open weights — see the model cards) beat generic models on
  Luganda. `Qwen/Qwen3-8B` is the cheap fallback.
- Corpora: `Sunbird/salt` (25k parallel sentences, CC-BY-SA-4.0, gated — needs `HF_TOKEN`),
  `kambale/luganda_prose` (~19.3M words, Apache-2.0), plus several alpaca-style sets that
  are **machine-translated** and stylistically unreliable.
- Recipe that works (from Sunbird's Sunflower paper): LoRA r=16 over all linear layers,
  4-bit base, response-only loss, then DPO on corrections at LR ≈ 5e-6.
- A frontier API model plus this prompt currently beats any small fine-tune. Do not
  regress real quality in the name of "having a trained model".
