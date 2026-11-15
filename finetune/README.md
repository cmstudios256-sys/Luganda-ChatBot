# Teaching Yogera to speak Luganda properly

The app alone gets you *reasonable* Luganda from a strong model plus a good prompt.
This directory is how you make the model belong to your users: their register, their
town, their greetings, their vocabulary — and how you prove, with numbers and native
speakers, that it improved.

```
  chat with users          thumbs-down + a native's rewrite
        │                             │
        ▼                             ▼
   /api/export  ──►  data/exported/{chats_sft,chats_dpo}.jsonl
        │                             │
        ▼                             ▼
  ingest_hf.py ─┐               train_dpo.py  (stage 2: what NOT to say)
  (public corpora) ──► build_dataset.py ──► out/{train,val,test}.jsonl
        │                             │
        ▼                             ▼
  validate_dataset.py  ─────►  train_lora.py  (stage 1, QLoRA, answer-tokens-only loss)
                                      │
                                      ▼
                              eval.py  +  human_rubric.csv  ──► ship or iterate
```

## Step 0 — get native speakers before you get GPUs

This is the step people skip, and it is the only one that decides quality.

- Recruit 2–3 native Luganda speakers who are comfortable writing it (not just speaking it).
- Every record in this repo ships `"verified": false`. Fix them, then flip the flag.
- Run `python finetune/eval.py --predictions <your model's answers>` and hand reviewers the
  generated `finetune/out/human_rubric.csv`. The column that matters is
  *"would a Muganda say this, y/n"* — aim for ≥80% yes before you call anything done.
- Pay them. This is linguistic labour, and it is the scarce input in every Ugandan-language model.

## Step 1 — data, and what each source is actually good for

| Source | Size | Licence | Use it for | Reality check |
|---|---|---|---|---|
| `finetune/data/seed_luganda.jsonl` (here) | 26 chats | yours | Format reference, style, register | Ours, unverified. A template, not a corpus. |
| `Sunbird/salt` (SALT) | 25k parallel sentences, 6 Ugandan languages | CC-BY-SA-4.0 | Translation SFT (both directions) | Gated: accept terms on the Hub, set `HF_TOKEN` |
| `Sunbird/Sunflower-14B`, `-32B` | Qwen3-based, 31 Ugandan languages | check model card | **The base model to fine-tune** | Already knows Luganda; you are adjusting voice |
| `kambale/luganda_prose` | 818k records / ~19.3M words | Apache-2.0 | Continued pre-training, tokenizer work | Monolingual prose, not dialogues |
| `kambale/luganda-english-parallel-corpus` | parallel | check card | Translation eval + SFT | Good references for chrF |
| `allandclive/alpaca_4k_luganda`, `saillab/alpaca-luganda-cleaned`, `ChrisToukmaji/luganda_instruction_tuning` | 1k–4k | check each card | Instruction-shaped warm start | **Machine-translated from Alpaca.** Expect stiff, English-shaped Luganda (`"square mita"`). Useful volume, unreliable style. |
| `CraneAILabs/pedagogy-luganda-reviewed` | education domain | check card | Domain SFT | Reviewed sets are rare — value them |
| Your own chats (`data/exported/`) | grows | yours | Style + preference pairs | The only data that matches your users |

```bash
# pull a starter set (works with no token; raw-file + parquet transports)
python finetune/ingest_hf.py --dataset allandclive/alpaca_4k_luganda --preset alpaca --limit 4000

# gated corpora need a token
export HF_TOKEN=hf_...
python finetune/ingest_hf.py --dataset Sunbird/salt --preset salt --limit 20000
```

Rule of thumb: **2,000–10,000 clean, native-verified conversational turns** is where a Luganda
chat adapter starts to feel real. 26 rows clones the formatting; it does not teach the language.

## Step 2 — validate, then build

```bash
python finetune/validate_dataset.py finetune/data            # schema, English leakage, geminates, dupes
python finetune/build_dataset.py --verified-only             # dedupe → 90/10/10 split, no leakage
```

The validator is Luganda-aware on purpose: it flags `amaji` (should be `amazzi`), `webale`
(should be `weebale`), bare `somero`/`nyumba` (missing geminates), the `ŋ` character, and turns
whose English function-word ratio suggests they are not Luganda at all. Warnings are your review
queue; errors block the build.

Output: `finetune/out/{train,val,test}.jsonl` in `{"messages": [...]}` chat format — the shape both
OpenAI fine-tuning and TRL/`SFTTrainer` expect — plus `stats.json` with vocabulary and length stats.

## Step 3 — train stage 1: QLoRA SFT (pick your route)

You do **not** need to own a GPU. Four equally valid ways to run stage 1, compared in
`../TRAINING-OPTIONS.md`:

| Route | Command | Hardware |
|---|---|---|
| Hosted fine-tune (no GPU) | `python finetune/train_openai.py` | none — uploads a JSONL |
| Free Colab / Kaggle T4 | `finetune/colab_qlora.ipynb` (uses `configs/lora_t4_8b.yaml`) | T4 16 GB |
| Rented A100 | `python finetune/train_lora.py --config finetune/configs/lora_sunflower14b.yaml` | A100 40 GB |
| Small model, no GPU | `train_lora.py --base-model Qwen/Qwen2.5-1.5B` | CPU, slow, weak |

Everything below describes the local QLoRA route; the data pipeline is identical for all of them.

```bash
python finetune/train_lora.py --config finetune/configs/lora_sunflower14b.yaml --dry-run  # no GPU, prints the plan
pip install torch transformers peft trl datasets accelerate bitsandbytes pyyaml
python finetune/train_lora.py --config finetune/configs/lora_sunflower14b.yaml             # needs the GPU
```

The recipe is the one Sunbird published for **Sunflower** (arXiv 2510.07203): LoRA **r=16**,
α=32 over all linear projections, **4-bit base weights**, and the loss computed **only on the
assistant's tokens** — which is what stops the model learning to echo instructions. That masking
is implemented in `build_response_collator()`; it is the part most tutorials get wrong.

Hardware, 4-bit QLoRA:

| Base | Params | VRAM (train) | Suggested card |
|---|---|---|---|
| `Qwen/Qwen3-8B` | 8B | ~16–20 GB | RTX 4090 / A10 24 GB |
| `Sunbird/Sunflower-14B` | 14B | ~22–28 GB | A100 40 GB / L40S (batch 2, seq 2048) |
| `Sunbird/Sunflower-32B` | 32B | ~40–48 GB | A100 80 GB / H100 |

On a CPU-only box you can still do everything except the actual optimisation step — the data
pipeline, the dry run, and the evaluation harness all run fine.

Why Sunflower-14B as the default: it already speaks 31 Ugandan languages, so you are teaching
**behaviour** (register, greetings, refusals, your product's voice), not vocabulary. With
Qwen3-8B you are teaching vocabulary *and* behaviour on a small dataset, and it shows.

## Step 4 — evaluate before you believe anything

```bash
python finetune/eval.py --judge                       # generates answers with the configured provider
python finetune/eval.py --predictions preds.jsonl     # or score any model's outputs
```

Three layers, in ascending order of trust:

1. **chrF++ / BLEU** against `data/eval/lug_eval_set.jsonl` (19 held-out items across greeting,
   register, translation, grammar, culture, safety, orthography). chrF++ ≥ 60 is a healthy
   starting point; it only tells you "close to the reference", not "good Luganda".
2. **LLM judge** on fluency / grammar / naturalness / culture-fit / language-purity. Cheap, A/B
   useful, do not put it on a slide as truth.
3. **`human_rubric.csv`** — 19 rows a native speaker fills in. This is the only number that
   decides whether you ship.

## Step 5 — train stage 2: DPO on human corrections

Every thumbs-down *with* a rewritten answer in the app is a `(prompt, chosen, rejected)` triple.

```bash
python finetune/train_dpo.py --pairs finetune/data/exported/chats_dpo.jsonl --dry-run
python finetune/train_dpo.py --pairs finetune/data/exported/chats_dpo.jsonl \
    --base Sunbird/Sunflower-14B --adapter finetune/out/yogera-sunflower14b-lora --beta 0.1
```

Sunflower used the same two-stage idea and reported that preference optimisation was what finally
killed the failure modes SFT could not: invented vocabulary, repetitive loops, English drift. Use a
small learning rate (5e-6) and a few hundred pairs minimum.

## Step 6 — serve it, and point the app at it

```bash
# merge the adapter into fp16 weights (merge_and_export: true in the config), then:
vllm serve finetune/out/yogera-sunflower14b-merged --served-model-name yogera-luganda --port 8000
```

```env
LLM_PROVIDER=compatible
LLM_BASE_URL=http://localhost:8000/v1
LLM_MODEL=yogera-luganda
```

That is the full circle: no API key, no data leaving your infrastructure, and a model tuned on the
Luganda your users actually write. Keep the export → build → train → eval loop running monthly;
`stats.json` and `eval_report.json` are your regression history.

## Failure modes to watch for

| Symptom | Cause | Fix |
|---|---|---|
| It starts sounding English-y after fine-tuning | Too high an LR, or English-heavy training rows | Drop LR to 5e-5, run validator, raise `--verified-only` |
| It repeats the user's question before answering | Loss computed on prompt tokens | Keep `train_on_responses_only: true` (already default here) |
| Invented words that "look" Luganda | Model filling gaps; expect it in low-resource settings | DPO on corrections; in-prompt rule "never invent a word, ask instead" |
| Register drifts to singular for elders | Your data is mostly peer-register | Balance the dataset; `register` field exists on every record for a reason |
| Great chrF, natives hate it | chrF rewards copying the reference's length and words | Human rubric is the real gate |
| Tokenizer eats geminates / apostrophes | Normalisation mismatch between training and serving | NFC-normalise both sides (the pipeline already NFC-normalises) |

## Licensing, briefly

SALT is **CC-BY-SA-4.0** (share-alike — it can infect derived data; read it before you ship
commercially). `luganda_prose` is Apache-2.0. Every alpaca-style set has its own card; several are
machine translations of Alpaca, which is itself subject to OpenAI's terms. Record what you used in
`out/manifest.json` (the build writes it), and keep the licences next to the weights.

## Sources worth reading

- Akera et al., *Sunflower: A New Approach To Expanding Coverage of African Languages in LLMs*,
  arXiv:2510.07203 — the LoRA r=16, response-only, DPO recipe this pipeline follows.
- Sunbird AI SALT corpus and the `Sunbird/*` model family on the Hugging Face Hub.
- `kambale/luganda_prose` corpus card — what a usable monolingual Luganda corpus looks like.
