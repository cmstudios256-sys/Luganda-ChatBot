# Yogera — a Luganda chatbot that answers like a Muganda

A working Luganda chat app (web UI + REST API), plus the fine-tuning pipeline that turns
real conversations into a model that actually speaks the language instead of translating into it.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  web/index.html          chat UI · streaming · thumbs up/down · correction box
├─────────────────────────────────────────────────────────────────────────┤
│  app/main.py             FastAPI: /api/chat (SSE), /api/chat/sync, /api/feedback,
│                          /api/export  →  conversations become training data
│  app/prompts.py          the Luganda system prompt: register, noun classes,
│                          orthography, greetings, refusal policy, few-shots
│  app/providers.py        OpenAI · Anthropic · Gemini · any OpenAI-compatible
│                          endpoint (vLLM/Ollama/OpenRouter) · offline demo
│  app/store.py            SQLite: conversations, messages, feedback, exports
├─────────────────────────────────────────────────────────────────────────┤
│  finetune/               ingest_hf → validate → build → train_lora (QLoRA)
│                          → train_dpo → eval (chrF++ / BLEU / judge / human sheet)
└─────────────────────────────────────────────────────────────────────────┘
```

![Yogera chat UI](docs/chat.png)

<table><tr>
<td><img src="docs/home.png" alt="empty state" width="380"></td>
<td><img src="docs/learning-loop.png" alt="learning drawer" width="380"></td>
</tr></table>

## 60-second start

```bash
cd yogera
pip install -r requirements.txt      # fastapi, uvicorn, httpx, pydantic
cp .env.example .env                 # then paste ONE api key into it
./run.sh                             # http://localhost:8000
```

No key? It still runs — a small offline demo replies in Luganda and the UI says **demo mode**,
so you can test the interface, the feedback loop and the export before you spend anything.

**Fastest way to a genuinely good Luganda model today:** point it at a strong frontier model,
not at a small fine-tune.

```env
# .env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4.1-mini
OPENAI_API_KEY=sk-...
```

Then, when you want the model to sound like *your* Luganda — the register, the greetings, the
vocabulary of your users — run the loop in `finetune/README.md`.

## Why it sounds native (and where it still doesn't)

The quality of a low-resource-language chatbot is mostly prompt and data engineering, so
`app/prompts.py` is a serious document, not three lines:

| Decision | Why it matters in Luganda |
|---|---|
| Default to Luganda **only**, never code-mixed | The usual failure is an English word slipping in where `obudde`, `essimu` or `obuyambi` exists |
| Register control (`peer` vs `respect`) | `Oli otya?` to an elder is rude; you need `Muli mutya?` and plural `-mu-` concords |
| Noun-class concord tables in the prompt | `abantu babiri` vs `ebintu bibiri` vs `ennyumba nnya` — this is what marks a native speaker |
| Greeting-by-time-of-day with correct answers | `Wasuze otya? → Nzuze bulungi`, `Osiibye otya? → Nsibye bulungi`, and the plural forms |
| Orthography rules (geminates, apostrophes, no `ŋ`) | `ssomero`, `ennyumba`, `w'omuntu` — doubled consonants change meaning (`okukola` vs `okukoola`) |
| "Never word-for-word translate English idiom" | Literal translation is the clearest tell of a non-native speaker |
| Refuse to invent facts, words, prices, dates | Hallucination is the documented weak spot of Ugandan-language models |
| Answer short, ask one clarifying question | Luganda chat is conversational; essays read as machine output |

It will still get things wrong. **Every seed record in `finetune/data/` is marked
`"verified": false`** on purpose: get two or three native speakers to go through them with the
`human_rubric.csv` sheet before you trust or train on any of it.

## API

| Method | Path | What it does |
|---|---|---|
| `POST` | `/api/chat` | Streaming reply (SSE: `meta`, `delta`, `error`, `done`) |
| `POST` | `/api/chat/sync` | Same thing, one JSON response |
| `POST` | `/api/feedback` | `{message_id, conversation_id, rating: 1|-1, corrected_text}` |
| `GET` | `/api/conversations[/{id}]` | History |
| `DELETE` | `/api/conversations/{id}` | Forget a thread |
| `POST` | `/api/export` | Write `finetune/data/exported/{chats_sft,chats_dpo}.jsonl` |
| `GET` | `/api/export/download` | Download the SFT file |
| `GET`/`POST` | `/api/settings` | Inspect or switch provider/model/key at runtime (memory only) |
| `GET` | `/api/health` | Provider status + usage stats |

```bash
curl -s localhost:8000/api/chat/sync -H 'content-type: application/json' \
  -d '{"message":"Osiibye otya nno?","register":"respect"}' | jq -r .reply
```

Request fields for `/api/chat`: `message`, `conversation_id?`, `language` (`lug`|`en`),
`tone` (`auto`|`peer`|`respect`), `teach_mode` (bool — adds an English gloss + grammar note),
`english_gloss` (bool), `temperature?`.

## Providers

Set one of these in `.env` (or live in the UI's settings drawer):

```env
LLM_PROVIDER=openai|anthropic|gemini|compatible|mock
LLM_MODEL=...
LLM_BASE_URL=...        # for compatible endpoints
LLM_TEMPERATURE=0.6
LLM_MAX_TOKENS=900
```

`compatible` is the interesting one: point it at a vLLM/Ollama server running Sunbird's
**Sunflower-14B** (`Sunbird/Sunflower-14B`) and you get a Luganda-first model with no data
leaving your machine. That is the end state of the fine-tuning work.

## Tests

```bash
pytest -q       # 14 tests, offline, ~1s: API, streaming, feedback→export, chrF/BLEU, dataset validation
```

## Layout

```
yogera/
├─ app/          FastAPI backend, Luganda prompts, providers, SQLite store
├─ web/          single-file chat UI (no CDN, no build step)
├─ finetune/     ingest · validate · build · train_lora · train_dpo · eval  (+ its own README)
├─ tests/
├─ .env.example  five provider recipes
├─ requirements.txt  Makefile  run.sh
```

## Honest limitations

- **Luganda is low-resource.** A frontier model with a good prompt is fluent-ish but still makes
  agreement errors; a small fine-tune on 1,000 rows will be *worse*, not better. The pipeline here
  is designed to grow into a real dataset, not to pretend 26 seed rows are one.
- **No retrieval yet.** There is no RAG layer, so it cannot answer questions about your business,
  prices or policies from documents. `store.py` and the export path are the seams where that goes.
- **Text only.** Luganda users overwhelmingly prefer voice. Sunbird ships usable
  `asr-whisper-large-v3-salt` and `tts-vits-lug` models — voice input is the highest-impact next step.
- **Demo mode is not a model.** With no key, replies are keyword-routed canned lines so the UI and
  the learning loop can be tested. It is labelled `demo` everywhere for that reason.

## Roadmap that would matter most

1. **Voice** — Whisper-Luganda for input, VITS-Luganda for output (both public on the Hub).
2. **WhatsApp/USSD front-end** — where Ugandan users actually are.
3. **RAG over your documents** — support and health content in Luganda, grounded, not guessed.
4. **Native-speaker review tool** — a small page that loads the review queue and writes
   `verified: true`; it is the bottleneck for every other improvement.

Licence: your call — the seed data is ours to give, the corpora you ingest are not (see
`finetune/README.md` for per-dataset licences).
