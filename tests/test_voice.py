"""Speech phase: text is made speakable before it is made audible, and nothing lies about it.

These tests never touch the network. The ``hf``/``http``/``local`` backends are exercised only
through their *failure* paths, which is the behaviour that matters most: a missing token or a
dead server must degrade to labelled silence, not a 500 and not a fake voice.
"""
from __future__ import annotations

import array
import json
import math
import wave

import pytest
from fastapi.testclient import TestClient

from app.main import app
from voice import check_takes, record_script, tts


def _write_wav(path, *, seconds=1.0, freq=220.0, amplitude=20000, rate=16000, channels=1):
    frames = int(rate * seconds)
    samples = array.array("h", (int(amplitude * math.sin(2 * math.pi * freq * i / rate))
                                for i in range(frames)))
    if channels == 2:
        stereo = array.array("h")
        for s in samples:
            stereo.extend((s, s))
        samples = stereo
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(channels)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes(samples.tobytes())
    return path


# --------------------------------------------------------------------------- #
# Making text speakable
# --------------------------------------------------------------------------- #
def test_markup_and_emoji_are_not_read_aloud():
    spoken, warnings = tts.normalize_for_speech("**Oli otya?** 🎉 Nsuubira oli bulungi.")
    assert spoken == "Oli otya?  Nsuubira oli bulungi." or spoken == "Oli otya? Nsuubira oli bulungi."
    assert "**" not in spoken and "🎉" not in spoken
    assert warnings == []


def test_small_numbers_are_spoken_in_luganda():
    spoken, warnings = tts.normalize_for_speech("Nkuwa emmere ya 3 ne ssente 10.")
    assert "ssatu" in spoken and "kkumi" in spoken
    assert "3" not in spoken
    assert warnings == []


def test_big_numbers_are_left_alone_and_reported():
    """Luganda numbers are irregular past ten; guessing is worse than saying '47'."""
    spoken, warnings = tts.normalize_for_speech("Abantu 47 bajja leero.")
    assert "47" in spoken
    assert any("47" in w for w in warnings)


def test_colon_numbers_are_untouched():
    spoken, _ = tts.normalize_for_speech("Ssaawa 10:30 kati.")
    assert "10:30" in spoken


def test_empty_text_produces_no_audio():
    result = tts.synthesize("   ")
    assert result.audio == b"" and result.content_type == "audio/wav"


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
def test_stub_is_a_valid_wav_and_admits_it():
    result = tts.synthesize("Oli otya mukwano gwange?", backend="stub")
    assert result.audio[:4] == b"RIFF"
    assert tts.duration_seconds(result.audio) > 0.5
    assert result.backend == "stub"
    assert result.is_real_voice is False
    assert any("not a voice" in w for w in result.warnings)


def test_stub_length_follows_the_text():
    short = tts.synthesize("Yee.", backend="stub").audio
    long = tts.synthesize(" ".join(["Olugero lwaffe lwali lulungi nnyo."] * 12), backend="stub").audio
    assert tts.duration_seconds(long) > tts.duration_seconds(short)


def test_missing_hf_token_degrades_instead_of_crashing(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)
    result = tts.synthesize("Weebale nnyo.", backend="hf")
    assert result.backend == "stub"
    assert any("HF_TOKEN" in w for w in result.warnings)


def test_unknown_backend_is_reported_not_silently_used():
    result = tts.synthesize("Tewali kizibu.", backend="telepathy")
    assert result.backend == "stub"
    assert any("unknown backend" in w for w in result.warnings)


def test_result_dict_hides_the_audio_bytes():
    d = tts.synthesize("Ka tugende.", backend="stub").as_dict()
    assert "audio" not in d and d["bytes"] > 0 and d["is_real_voice"] is False


# --------------------------------------------------------------------------- #
# HTTP surface
# --------------------------------------------------------------------------- #
@pytest.fixture()
def client(isolated_db):
    return TestClient(app)


def test_tts_status_is_honest(client):
    body = client.get("/api/tts/status").json()
    assert body["backend"] in {"stub", "hf", "http", "local"}
    assert body["real_voice"] is (body["backend"] != "stub")
    assert "hf" in body["backends"]


def test_tts_endpoint_returns_audio_with_a_backend_header(client):
    resp = client.post("/api/tts", json={"text": "Oli otya?", "backend": "stub"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("audio/")
    assert resp.headers["x-tts-backend"] == "stub"
    assert resp.headers["x-tts-real-voice"] == "false"
    assert resp.content[:4] == b"RIFF"


def test_tts_headers_survive_luganda_punctuation(client):
    """A curly quote or an em dash used to be a 500 because HTTP headers are latin-1."""
    resp = client.post("/api/tts", json={"text": "“Ekyo” — kituufu.", "backend": "stub"})
    assert resp.status_code == 200
    resp.headers["x-tts-warnings"].encode("latin-1")          # would raise if unsanitised


def test_tts_rejects_empty_text(client):
    assert client.post("/api/tts", json={"text": ""}).status_code == 422


# --------------------------------------------------------------------------- #
# Recording script
# --------------------------------------------------------------------------- #
def test_script_spreads_across_lengths_and_keeps_reading_order():
    lines = [f"{'ekigambo ' * n}wansi." for n in range(3, 40)]
    picked = record_script.select(lines, 10)
    assert len(picked) == 10
    # the same lines get chosen regardless of input order; only the reading order may differ
    assert set(picked) == set(record_script.select(list(reversed(lines)), 10))
    assert {len(p.split()) for p in picked} != {5}, "must not all be the same length"


def test_script_manifest_is_numbered_and_escaped():
    page, manifest = record_script.build(["Oli otya?", "Weebale <nnyo>."], minutes=None)
    assert [m["id"] for m in manifest] == ["0001", "0002"]
    assert all(m["est_seconds"] > 0 for m in manifest)
    assert "&lt;nnyo&gt;" in page and "<nnyo>" not in page
    assert "consent" in page.lower(), "the sheet must carry the consent warning"


def test_script_pulls_only_signed_off_text_when_asked(monkeypatch):
    fake = [{"text": "Ebbaluwa eno ntuufu era nnungi nnyo.", "verified": True},
            {"text": "Kino tekinnakakasibwa era kiyinza okuba ekikyamu.", "verified": False}]
    monkeypatch.setattr(record_script, "load_segments", lambda: (fake, {}))
    assert record_script.sentences_from_corpus(verified_only=True) == [
        "Ebbaluwa eno ntuufu era nnungi nnyo."]


def test_script_skips_fragments_too_short_to_pronounce(monkeypatch):
    fake = [{"text": "Yee. Nedda nnyo kubanga abantu bangi bajja leero ku makya.", "verified": True}]
    monkeypatch.setattr(record_script, "load_segments", lambda: (fake, {}))
    lines = record_script.sentences_from_corpus(verified_only=False)
    assert "Yee." not in lines


# --------------------------------------------------------------------------- #
# Checking a recording session
# --------------------------------------------------------------------------- #
def _session(tmp_path, *, seconds=2.0, amplitude=20000, lines=3):
    manifest = [{"id": f"{i:04d}", "file": f"{i:04d}.wav", "text": "Oli otya mukwano.",
                 "words": 3, "est_seconds": 2.0} for i in range(1, lines + 1)]
    (tmp_path / "manifest.jsonl").write_text(
        "\n".join(json.dumps(m) for m in manifest) + "\n", encoding="utf-8")
    for m in manifest:
        _write_wav(tmp_path / m["file"], seconds=seconds, amplitude=amplitude)
    return tmp_path


def test_clean_session_passes(tmp_path):
    problems, stats = check_takes.check(_session(tmp_path))
    assert problems == []
    assert stats["files"] == 3 and stats["missing"] == 0


def test_silent_take_is_caught(tmp_path):
    problems, _ = check_takes.check(_session(tmp_path, amplitude=0))
    assert any("mostly silence" in p for p in problems)


def test_clipping_is_caught(tmp_path):
    problems, _ = check_takes.check(_session(tmp_path, amplitude=32767))
    assert any("clipped" in p for p in problems)


def test_missing_line_is_named(tmp_path):
    _session(tmp_path, lines=3)
    (tmp_path / "0002.wav").unlink()
    problems, stats = check_takes.check(tmp_path)
    assert stats["missing"] == 1
    assert any("missing audio for line 0002" in p for p in problems)


def test_unreadable_wav_is_not_passed_off_as_fine(tmp_path):
    (tmp_path / "0001.wav").write_bytes(b"not a wav at all")
    problems, _ = check_takes.check(tmp_path)
    assert any("not a readable WAV" in p for p in problems)
