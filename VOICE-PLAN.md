# Text first, voice second

Your instinct is right, and it is right for a reason worth writing down: **speech inherits
every flaw in the text and adds new ones you cannot debug.** If the sentence is wrong, the
recording of it is wrong in a more convincing way. If the sentence is right, a merely
acceptable voice still sounds better than a perfect voice reading bad Luganda.

So the order is:

```
1. text that a Muganda accepts as a normal reply     <- the work, and where time goes
2. that same text spoken aloud                       <- a swap of one setting, later
3. speech coming in (voice notes)                    <- cheap on top of step 2
```

This is not a delay tactic. Step 2 can be heard *this week* with an existing Luganda
TTS model (see "Today, for free" below) — what you cannot rush is step 1, and a cloned
voice bolted onto a shaky text model just produces fluent-sounding nonsense, which is the
worst possible product: it sounds like it understands.

---

## Phase 1 — "conversation level", defined so it can be failed

"Good enough to talk to" is not a test. This is the gate I will hold the project to:

**The probe set.** ~30 short exchanges, written as a Kampala chat would actually go, not as
a textbook would: greeting + how are you, asking the time and a date, directions, price and
haggling, school/fees, weather, a small errand, asking it to repeat or explain in simpler
words, switching to English mid-thread and back, an instruction ("write a short letter to my
friend"), a correction ("no, that's not what I meant"), and two turn-taking checks ("what did
I just ask you?"). Drafted in `finetune/data/conversation_probes.jsonl` — **correct my
Luganda in those probes before using them**; a wrong probe measures the wrong thing.

**How it is scored.** Three native speakers, blind (they never see whether a reply came from
the prompt+retrieval baseline or a fine-tune), 1–5 on five axes:

| Axis | 1 means | 5 means |
|---|---|---|
| Would a Muganda say this? | "a learner wrote it" | "my cousin's WhatsApp message" |
| Grammar & concord | wrong noun-class agreement | nothing to correct |
| Answers the actual question | evades, or answers a different question | answers it, briefly |
| Register/tone | teacher-to-pupil, or pulpit | fits how a friend talks to you |
| Trouble when you push back | collapses into English | stays in Luganda, repairs gracefully |

**Pass bar:** mean ≥ 4.0 on the first axis, no axis below 3.5, and fewer than 10% of replies
need a rewrite. Anything less, we keep working on text — a fine-tune that scores 3.6 is not
"almost there", it is the baseline with extra maintenance.

**Free automated checks, on every change** (they catch regressions a human rater is too nice
to report): English-word leakage per reply, geminate mangling (`ss`/`nn` preserved exactly),
clock-format compliance (`ssaawa emu ey'oku makya`, never `7:00 AM`), and agreement spot-checks
on a fixed list of noun-class pairs.

**What actually moves the needle here, in order:** verified conversational text (the review
sheet), retrieval of your own verified letters and stories, few-shots in the style you want,
then a fine-tune *only if* it beats the prompt baseline on the probe set. That last ordering
is the whole of `TRAINING-OPTIONS.md` and it does not change for voice.

---

## Phase 2 — speaking, with a voice that is really Luganda

### Today, for free

Sunbird publishes `tts-vits-lug` — a real Luganda voice. It is one speaker, it will not
sound like anyone you know, and it does not do emotion. It is enough to hear your bot talk
and to decide whether the *text* is good enough to bother cloning.

```bash
export YOGERA_TTS=hf HF_TOKEN=hf_…            # Sunbird/tts-vits-lug, no GPU needed
make run
curl -X POST localhost:8000/api/tts -H 'content-type: application/json' \
     -d '{"text":"Oli otya? Nze Yogera."}' --output reply.wav
```

This is wired already: `voice/tts.py` + `POST /api/tts` + `GET /api/tts/status`. With no
backend configured the endpoint returns **silence with a header saying it is silence** —
`X-TTS-Real-Voice: false` — because an endpoint that quietly fakes a voice is how demos lie.

### Then: your voice, or a native speaker's you have permission to use

The catch nobody mentions: a *cloned* voice needs 1–3 hours of clean speech from one person,
recorded on purpose. `python3 -m voice.record_script` builds the script (numbered lines, no
line longer than a breath, coverage of short/long, questions, statements, and the `ss`/`nn`
sounds the language leans on), and `python3 -m voice.check_takes voice/out/` inspects the
result before it costs GPU time: silent takes, clipped takes, read-too-fast takes, missing
numbers.

Honest quantity expectations for a single-speaker Luganda voice:

| Audio | What you get |
|---|---|
| 10–20 min | demo-quality clone. Distinctive but wobbly, wrong on rare words. Fine to show, not to ship |
| 1–2 h | usable assistant voice. Most sentences clean, occasional stumble on names/numbers |
| 5 h+ | good. Prosody starts carrying meaning — questions sound like questions |
| ~10 h of *conversational* speech | the only route to a voice that sounds like a person thinking, not reading |

Two routes once you have the audio:

* **Fine-tune a new-language TTS** — XTTS-v2 (17 languages; Luganda is not one, but the
  community routine for adding a language is a 1–3 h fine-tune) or a VITS/Piper voice.
  This is GPU work: a rented T4 for a few hours, or Colab, same shape as the LLM story in
  `TRAINING-OPTIONS.md`. Inference afterwards can run on CPU (Piper does), which is what you
  want for a web app.
* **Hosted cloning** — a vendor that supports Luganda. Check the language list first; most
  do not list it, and a "close enough" Bantu accent is instantly wrong to a Muganda ear.

Naming: whoever lends the voice signs a written consent covering purpose, revocation, and
"will not be made to say things they did not say". Recording a TV presenter, a TikToker or a
preacher off existing audio is the one thing in this plan that cannot be undone — Luganda has
no deepfake detectors, so the harm lands on real people with no recourse.

### Voice changes what the bot should say

Short sentences, no bullet lists, no digits (`147` must be spoken as words — the pipeline
already expands 0–100 and *refuses to guess* past that, reporting instead), and one question
at a time. A voice-mode prompt flag is the next small piece of work, not a rewrite: same
model, stricter length.

---

## Phase 3 — hearing you (voice notes)

Worth doing early even though it is "speech", because it is cheap and it is how people
actually message: WhatsApp voice notes, hands-free, low literacy friction. Sunbird's
`asr-whisper-large-v3-salt` and `asr-wav2vec2-lug` both exist. Two guards: **WER is measured on
your own recordings, not trusted from a model card**, and short clips must be rejected rather
than transcribed confidently — Whisper-family models hallucinate fluent text on near-silence,
and a hallucinated *Luganda* sentence is indistinguishable from a real one to you in review.
Browser side is a single `MediaRecorder` and a `POST` of raw audio; no new dependencies.

---

## The order, with gates

| Step | What it unlocks | Gate to pass before moving on | Your time |
|---|---|---|---|
| 1. Review sheet on real text | the bot can train at all | ~5,000+ verified words, mixed registers | ~1 h |
| 2. Bible + letters ingested | volume without review | verified corpus ≥ 100k words | minutes |
| 3. Conversation probes, 3 raters | a number for "conversation level" | mean ≥ 4.0, no axis < 3.5 | ~2 h |
| 4. Hear it: hosted Luganda TTS | the demo everyone asks for | replies are short enough to speak | ~10 min |
| 5. Record a voice (script + checks) | your own voice, later | ≥ 1 h clean audio, checks green | 3–4 h |
| 6. TTS fine-tune (rented T4) | the voice in the product | blind A/B vs hosted voice, natives prefer yours | 1 day |
| 7. ASR in, voice notes | hands-free use | WER measured on 50 of your own recordings | 1 day |

Steps 1–4 are all doable without a GPU and without spending money. Step 5 costs only your
afternoon and a quiet room. Only 6 and 7 need hardware, and by then you will know whether
the text is worth it — which is exactly the point of doing it in this order.

**What I will not do:** ship a "voice" that is silence with a nice label, or call a
text-to-speech model a native speaker. The endpoint carries its backend in a header so
nobody downstream can mistake one for the other.
