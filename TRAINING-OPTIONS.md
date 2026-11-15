# Six ways to "train" Yogera without owning a GPU

Fine-tuning weights is only one of six options, and for a Luganda bot it is rarely the
first one you should spend money on. Ordered by cost and by how fast you see a result.

| # | Route | Needs a GPU? | Cost | Time to first result | What it actually buys you |
|---|---|---|---|---|---|
| 1 | **Retrieval + exemplar memory** (built in) | no | $0 | minutes | Your facts, your voice, your corrections — live in the next reply |
| 2 | **Hosted fine-tuning API** (`train_openai.py`) | no | a few $ per run | ~1 hour | A model that behaves like your data, on someone else's GPUs |
| 3 | **Free GPU notebook** (`colab_qlora.ipynb`) | free T4 | $0 | 1–2 h | An 8B adapter you can run on a laptop afterwards |
| 4 | **Rent a GPU by the hour** (`train_lora.py`) | yes, rented | ~$1–5 per run | 1–3 h | Sunflower-14B quality, kept entirely in-house |
| 5 | **CPU-only LoRA** on a small model | no | $0 | 6–48 h | Mostly a learning exercise; weak Luganda |
| 6 | **Continued pre-training** on Luganda prose | yes, serious | $100s–$1000s | days | A real language model, not a chatbot |

---

## Route 1 — don't train at all: make it learn from memory (already implemented)

This is the honest answer to "can I train without a GPU". Weights stay frozen; behaviour
changes immediately.

- `app/knowledge.py` — your documents go into SQLite FTS5, retrieved with BM25 and injected
  into the system prompt. No vector DB, no dependencies, works on a laptop.
- Every 👍 creates a **preferred-phrasing exemplar**; every 👎 with a rewrite creates one
  from the human's version. Similar questions retrieve those phrasings, so the bot reuses
  what a Muganda approved instead of inventing a new sentence.
- A small Luganda↔English glossary bridges queries: ask `ssente mmeka ku mmwanyi?` and the
  English FAQ that says "coffee price" still matches.

```bash
curl -X POST localhost:8000/api/knowledge -H 'content-type: application/json' -d '{
  "title": "Yogera Coffee — prices and delivery",
  "text": "A 500g bag costs 18,000 UGX, a 1kg bag 34,000 UGX. Delivery in Kampala is 5,000 UGX and takes one day."
}'
curl "localhost:8000/api/knowledge/search?q=ssente%20mmeka%20ku%20mmwanyi%3F"
```

**Limits, stated plainly:** it cannot teach new *grammar*, only new facts and phrasings.
Retrieval is keyword-based, so it fails when users paraphrase hard. But it fixes the two
things users actually complain about — "it doesn't know our prices" and "it doesn't sound
like us" — for free, today. Start here.

## Route 2 — hosted fine-tuning (no GPU, no ops)

Your machine only uploads a JSONL and polls a job. `finetune/out/train.jsonl` is already in
the shape these APIs want.

```bash
python finetune/train_openai.py --dry-run     # validates the file locally, costs nothing
python finetune/train_openai.py               # uploads + starts the job
python finetune/train_openai.py --status ftjob-...
```

Providers that take the same file format, with the trade-offs that matter:

| Provider | Notes |
|---|---|
| OpenAI (`gpt-4.1-mini`, `gpt-4o-mini` classes) | Simplest path; strong base Luganda; per-token training + higher inference price |
| Together AI / Fireworks | Fine-tune open models (Llama/Qwen) into LoRA adapters you can also download |
| Predibase / OpenPipe | LoRA specialists, generous cheap tiers |
| Google (Gemini / Vertex tuning) | Good if you are already on GCP |
| Azure OpenAI | Same as OpenAI with enterprise billing/region controls |

**Judge the result, don't assume:** fine-tuning a strong base on 1,000 rows usually makes it
*sound more like your data* and slightly *worse at everything else*. Run
`finetune/eval.py` before and after and keep whichever wins.

## Route 3 — free GPU notebooks

`finetune/colab_qlora.ipynb` is ready: installs the stack, builds the dataset, trains
**Qwen3-8B** QLoRA in 4-bit with a T4-safe config (`configs/lora_t4_8b.yaml`, fp16 not bf16,
batch 1 + grad-accum 16, 1024-token sequences), then converts to GGUF Q4_K_M and tells you how
to serve it with Ollama.

| Free tier | GPU | Caveat |
|---|---|---|
| Google Colab (free) | T4 16 GB | No bf16; ~12 h limit; disconnects |
| Colab Pro | L4 / A100 | ~$10–50/mo; the A100 can take Sunflower-14B |
| Kaggle | 2× T4 or P100, ~30 h/week | Best free option for a 7–8B QLoRA |
| Lightning AI | monthly free credits | Persistent environment, less setup pain |
| SageMaker Studio Lab | T4, 4 h sessions | Free account approval required |

A T4 **cannot** train Sunflower-14B — not "slowly", it simply will not fit. Don't waste the
afternoon; use the 8B config or go to route 4.

## Route 4 — rent the GPU for an hour or two

This is what `configs/lora_sunflower14b.yaml` is for. Rent by the second, train, download the
adapter, destroy the pod.

| Service | Typical card | Roughly |
|---|---|---|
| RunPod | A100 40 GB / L40S | ~$1–2 per hour |
| Lambda Labs | A100 / H100 | ~$1.5–3 per hour |
| Vast.ai | community A100 | cheapest, variable reliability |
| Hugging Face Jobs / Endpoints | A10G / A100 | billed hourly, easy to script from the repo |

A 2,000-row Luganda SFT run on Sunflower-14B at 4-bit is roughly 1–3 hours: **a $3–6
experiment**. That is the cheapest way to get the *best* Luganda, and no data leaves your
control. Steps are in `finetune/README.md`.

## Route 5 — CPU-only LoRA (possible, mostly educational)

No GPU at all: `peft` on CPU, or `llama.cpp`'s LoRA training on a GGUF.

- Realistic ceiling: **0.5B–3B** parameters, a few hundred short examples, 6–48 hours.
- Quality: it will learn your *formatting and a handful of phrases*. It will not learn Luganda
  grammar; a 1B model prompted well beats a 1B model tuned badly.
- Where it is genuinely useful: teaching a tiny local model a fixed set of intents for offline
  kiosks/USSD, where you cannot call an API at all.

If you want to try it anyway, keep it honest:
```bash
pip install torch transformers peft datasets
python finetune/train_lora.py --config finetune/configs/lora_qwen3_8b.yaml \
       --base-model Qwen/Qwen2.5-1.5B --dry-run     # then drop --dry-run and be patient
```

## Route 6 — continued pre-training (only if you are building a Luganda LM)

Something like Sunflower, not a chatbot. `kambale/luganda_prose` (~19.3M words) plus SALT gets
you a tokenizer and a base model; you then need SFT and DPO on top. Plan weeks, a rented
multi-GPU node, and a proper evaluation suite. Sunbird published exactly this recipe
(arXiv:2510.07203) — read it before spending anything.

---

## What I would actually do, in order

1. **Today, free:** add 20–50 documents to the knowledge base (prices, policies, hours,
   common questions), chat in Luganda, and 👎/correct the answers. Route 1 makes the bot yours
   without a single training run.
2. **When the corrections pile up (300+):** Route 2 — a hosted fine-tune on `gpt-4.1-mini`
   costs a few dollars, needs no GPU, and shows you whether your data is good enough.
3. **If the fine-tune helps:** Route 3 or 4 — train on Sunflower-14B and serve it yourself.
   That is the end state: no API bill, no data leaving your infra, Luganda-first model.
4. **Only ever:** compare before/after with `finetune/eval.py` and a native-speaker rubric.
   "It feels better" after an hour of tweaking is how projects ship regressions.

## Three truths that save money

- **Data beats compute.** 2,000 verified turns on an 8B model beats 200,000 scraped turns on a
  32B model. The bottleneck is native review, not FLOPS.
- **You cannot fine-tune facts.** Prices, stock, opening hours, policy changes — those live in
  retrieval (Route 1). Fine-tuning is for *style, register and behaviour*.
- **Pricing moves.** Every number above is an order of magnitude, not a quote. Check the
  provider's page before you launch a job; this file is documentation, not a contract.
