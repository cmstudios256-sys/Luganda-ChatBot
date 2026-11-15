"""Turning a reply into audio — with the honest gaps labelled.

The order of work in this project is text first, voice second, and this module exists to
make the second step a swap rather than a rewrite:

    reply text  ->  normalize_for_speech()  ->  a backend  ->  audio bytes

Backends, cheapest first:

* ``stub``   — silence of roughly the right length. Not a voice. It exists so the endpoint,
  the UI and the tests can be built and verified *today*, without pretending we have audio.
* ``hf``     — Sunbird's ``tts-vits-lug`` on the HuggingFace inference API. A real Luganda
  voice you can hear this afternoon, if the model answers for your token.
* ``http``   — anything you run that takes JSON and returns audio (XTTS-v2 server, Piper HTTP,
  your own VITS). This is where a *cloned native voice* plugs in later.
* ``local``  — a shell command template, e.g. ``piper --model luganda.onnx --output-file {out}``.

Nothing here is allowed to pretend: every response says which backend produced it, and the
stub never claims to be speech.
"""
from __future__ import annotations

import io
import os
import re
import shlex
import struct
import subprocess
import wave
from dataclasses import dataclass, asdict

# Luganda cardinal numbers, 0-10 plus the tens. DRAFT — a native speaker should confirm this
# table before it reads anything important out loud; see the note in normalize_for_speech().
NUMBERS = {
    "0": "zeero", "1": "emu", "2": "bbiri", "3": "ssatu", "4": "nnya", "5": "ttaano",
    "6": "mukaaga", "7": "musanvu", "8": "munaana", "9": "mwenda", "10": "kkumi",
    "20": "amakumi abiri", "30": "amakumi asatu", "40": "amakumi ana",
    "50": "amakumi ataano", "60": "amakumi mukaaga", "70": "amakumi musanvu",
    "80": "amakumi munaana", "90": "amakumi mwenda", "100": "kikumi",
}

_MARKUP_RE = re.compile(r"(\*\*|__|`|#+\s*|^\s*[-*]\s+)", re.MULTILINE)
_EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F]")
_NUMBER_RE = re.compile(r"(?<![\w:])(\d{1,3})(?![\w:])")


def normalize_for_speech(text: str) -> tuple[str, list[str]]:
    """Make written Luganda speakable. Returns (text, warnings).

    Two jobs only, because both are things a TTS engine will get wrong:

    1. **Strip markup and emoji** — "**Oli otya?**" must not be read as "asterisk asterisk".
    2. **Spell out numbers it is safe to spell out.** Luganda numbers are irregular past ten
       (11 is ``kkumi n'emu``, 21 is not simply concatenated), so anything outside the table
       is left as digits and *reported* instead of guessed. A wrong number read aloud is
       worse than a number read in English.
    """
    warnings: list[str] = []
    out = _EMOJI_RE.sub("", _MARKUP_RE.sub("", text))

    def repl(m: re.Match[str]) -> str:
        digits = m.group(1)
        if digits in NUMBERS:
            return NUMBERS[digits]
        warnings.append(f"left as digits: {digits} (no confident Luganda form)")
        return digits

    out = _NUMBER_RE.sub(repl, out)
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\n{2,}", "\n", out).strip()
    return out, warnings


def stub_wav(text: str, *, sample_rate: int = 16000) -> bytes:
    """Silence, length-matched to the text. A placeholder, clearly not a voice."""
    seconds = max(0.6, min(30.0, 0.12 + 0.32 * len(text.split())))
    frames = int(sample_rate * seconds)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(b"\x00\x00" * frames)
    return buf.getvalue()


@dataclass
class SpeechResult:
    audio: bytes
    backend: str
    content_type: str
    text: str
    warnings: list[str]

    @property
    def is_real_voice(self) -> bool:
        return self.backend != "stub"

    def as_dict(self) -> dict:
        d = asdict(self)
        d.pop("audio")
        d["bytes"] = len(self.audio)
        d["is_real_voice"] = self.is_real_voice
        return d


def backend_name() -> str:
    return (os.environ.get("YOGERA_TTS") or "stub").strip().lower()


def synthesize(text: str, *, backend: str | None = None) -> SpeechResult:
    """Speak ``text``. Never raises for a missing backend — it degrades to stub + a warning."""
    backend = (backend or backend_name()).lower()
    spoken, warnings = normalize_for_speech(text)
    if not spoken:
        return SpeechResult(b"", "none", "audio/wav", "", ["nothing to speak"])

    if backend == "stub":
        warnings.append("stub backend: silence, not a voice — set YOGERA_TTS to use a real one")
        return SpeechResult(stub_wav(spoken), "stub", "audio/wav", spoken, warnings)

    try:
        if backend == "hf":
            return _hf(spoken, warnings)
        if backend == "http":
            return _http(spoken, warnings)
        if backend == "local":
            return _local(spoken, warnings)
    except Exception as exc:                                     # noqa: BLE001
        warnings.append(f"{backend} backend failed ({type(exc).__name__}: {exc}) — "
                        "falling back to silence")
        return SpeechResult(stub_wav(spoken), "stub", "audio/wav", spoken, warnings)

    warnings.append(f"unknown backend {backend!r}")
    return SpeechResult(stub_wav(spoken), "stub", "audio/wav", spoken, warnings)


def _hf(spoken: str, warnings: list[str]) -> SpeechResult:
    import httpx

    model = os.environ.get("YOGERA_TTS_MODEL", "Sunbird/tts-vits-lug")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
    if not token:
        raise RuntimeError("HF_TOKEN is not set")
    resp = httpx.post(
        f"https://api-inference.huggingface.co/models/{model}",
        headers={"Authorization": f"Bearer {token}"},
        json={"inputs": spoken},
        timeout=120.0,
    )
    resp.raise_for_status()
    ctype = resp.headers.get("content-type", "audio/flac").split(";")[0]
    if not ctype.startswith("audio/"):
        raise RuntimeError(f"model returned {ctype}: {resp.text[:200]}")
    return SpeechResult(resp.content, "hf", ctype, spoken, warnings)


def _http(spoken: str, warnings: list[str]) -> SpeechResult:
    import httpx

    url = os.environ.get("YOGERA_TTS_URL")
    if not url:
        raise RuntimeError("YOGERA_TTS_URL is not set")
    payload = {"text": spoken, "language": "lug", "speaker": os.environ.get("YOGERA_TTS_SPEAKER", "")}
    resp = httpx.post(url, json=payload, timeout=180.0)
    resp.raise_for_status()
    ctype = resp.headers.get("content-type", "audio/wav").split(";")[0]
    return SpeechResult(resp.content, "http", ctype, spoken, warnings)


def _local(spoken: str, warnings: list[str]) -> SpeechResult:
    import tempfile

    template = os.environ.get("YOGERA_TTS_CMD")
    if not template:
        raise RuntimeError("YOGERA_TTS_CMD is not set")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out_path = tmp.name
    cmd = template.format(text=shlex.quote(spoken), out=shlex.quote(out_path))
    subprocess.run(shlex.split(cmd) if "{text}" not in template else cmd,
                   shell="{" in template, check=True, capture_output=True)
    with open(out_path, "rb") as fh:
        audio = fh.read()
    os.unlink(out_path)
    return SpeechResult(audio, "local", "audio/wav", spoken, warnings)


def duration_seconds(wav_bytes: bytes) -> float:
    """Length of a WAV, for the tests and the recording script. 0.0 for anything else."""
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as fh:
            return fh.getnframes() / float(fh.getframerate())
    except (wave.Error, EOFError, struct.error):
        return 0.0
