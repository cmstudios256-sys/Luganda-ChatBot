"""End-to-end tests for the Yogera API and the data pipeline.

Run: pytest -q
Everything runs offline in demo mode (no API key, no network).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session", autouse=True)
def isolated_db():
    """Point the app at a throwaway SQLite file before it is imported."""
    tmp = tempfile.mkdtemp()
    os.environ["YOGERA_DB"] = str(Path(tmp) / "test.db")
    os.environ["LLM_PROVIDER"] = "mock"          # deterministic, offline
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "LLM_API_KEY"):
        os.environ.pop(key, None)
    sys.path.insert(0, str(ROOT))
    yield


@pytest.fixture(scope="session")
def client(isolated_db):
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from app.main import app  # noqa: PLC0415

    return TestClient(app)


def test_health_reports_demo_mode(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["status"]["demo_mode"] is True
    assert body["status"]["provider"] == "mock"


def test_index_serves_the_ui(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Yogera" in r.text
    assert "text/event-stream" not in r.headers.get("content-type", "")


def test_sync_chat_answers_in_luganda_and_persists(client):
    r = client.post("/api/chat/sync", json={"message": "Osiibye otya nno?"})
    assert r.status_code == 200
    body = r.json()
    assert body["conversation_id"].startswith("c_")
    assert "Nsibye" in body["reply"]
    conv = client.get(f"/api/conversations/{body['conversation_id']}").json()
    roles = [m["role"] for m in conv["messages"]]
    assert roles == ["user", "assistant"]


def test_streaming_chat_emits_meta_delta_done(client):
    r = client.post("/api/chat", json={"message": "Oli otya?"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = [line.split(":", 1)[1].strip() for line in r.text.splitlines() if line.startswith("event:")]
    assert events[0] == "meta"
    assert "done" in events
    assert "delta" in events


def test_conversation_history_is_reused(client):
    first = client.post("/api/chat/sync", json={"message": "Nze Sarah."}).json()
    second = client.post(
        "/api/chat/sync",
        json={"message": "Njagala okuyiga Oluganda.", "conversation_id": first["conversation_id"]},
    ).json()
    assert second["conversation_id"] == first["conversation_id"]
    conv = client.get(f"/api/conversations/{first['conversation_id']}").json()
    assert len(conv["messages"]) == 4


def test_unknown_conversation_is_404(client):
    r = client.post("/api/chat/sync", json={"message": "Oli otya?", "conversation_id": "c_nope"})
    assert r.status_code == 404


def test_feedback_rejects_bad_rating(client):
    conv = client.post("/api/chat/sync", json={"message": "Weebale."}).json()
    r = client.post(
        "/api/feedback",
        json={"message_id": conv["message_id"], "conversation_id": conv["conversation_id"], "rating": 5},
    )
    assert r.status_code == 400


def test_correction_flows_into_export(client):
    conv = client.post("/api/chat/sync", json={"message": "Oli otya?"}).json()
    client.post(
        "/api/feedback",
        json={
            "message_id": conv["message_id"],
            "conversation_id": conv["conversation_id"],
            "rating": -1,
            "corrected_text": "Gyendi, weebale. Oli otya ggwe?",
        },
    )
    r = client.post("/api/export")
    assert r.status_code == 200
    body = r.json()
    assert body["preference_pairs"] >= 1
    dpo = Path(body["dpo_path"]).read_text(encoding="utf-8").strip().splitlines()
    rec = json.loads(dpo[0])
    assert rec["chosen"] == "Gyendi, weebale. Oli otya ggwe?"
    assert rec["prompt"] == "Oli otya?"


def test_settings_endpoint_switches_provider(client):
    r = client.post("/api/settings", json={"provider": "mock", "temperature": 0.3})
    assert r.status_code == 200
    assert r.json()["provider"] == "mock"


def test_delete_conversation(client):
    conv = client.post("/api/chat/sync", json={"message": "Kale."}).json()
    assert client.delete(f"/api/conversations/{conv['conversation_id']}").json()["deleted"] is True
    assert client.get(f"/api/conversations/{conv['conversation_id']}").status_code == 404


# --------------------------------------------------------------------------- #
# Prompt + pipeline units
# --------------------------------------------------------------------------- #
def test_system_prompt_contains_the_language_rules():
    from app.prompts import build_system_prompt  # noqa: PLC0415

    prompt = build_system_prompt(register="respect", teach_mode=True, english_gloss=True)
    assert "Oluganda" in prompt
    assert "noun-class" in prompt.lower() or "Noun classes" in prompt
    for marker in ("Ssebo", "Wasuze otya", "weebale"):
        assert marker in prompt
    assert "respectful register" in prompt      # register note injected
    assert "learning Luganda" in prompt         # teach note injected


def test_chrf_and_bleu_reward_matches():
    sys.path.insert(0, str(ROOT / "finetune"))
    from eval_lib import bleu, chrf_plus_plus  # noqa: PLC0415

    ref = "Abaana bagenda ku ssomero."
    assert chrf_plus_plus(ref, ref) == pytest.approx(100.0)
    assert chrf_plus_plus("I do not know.", ref) < 30
    scores = bleu([(ref, ref)])
    assert scores["bleu"] > 80
    assert bleu([("hello", "Abaana bagenda ku ssomero.")])["bleu"] < 10


def test_dataset_validation_catches_english_and_missing_target(tmp_path):
    sys.path.insert(0, str(ROOT / "finetune"))
    from validate_dataset import check_record  # noqa: PLC0415

    good = {"messages": [{"role": "user", "content": "Oli otya?"},
                         {"role": "assistant", "content": "Gyendi, weebale."}]}
    errors, _ = check_record(good)
    assert errors == []

    no_target = {"messages": [{"role": "user", "content": "Oli otya?"}]}
    assert any("assistant" in e for e in check_record(no_target)[0])

    english = {"messages": [{"role": "user", "content": "How are you doing today my friend?"},
                            {"role": "assistant", "content": "I am fine and you are the best of all my friends today"}]}
    assert any("English" in w for w in check_record(english)[1])


def test_build_dataset_splits_and_dedupes(tmp_path):
    sys.path.insert(0, str(ROOT / "finetune"))
    import build_dataset  # noqa: PLC0415

    rows = [
        {"messages": [{"role": "user", "content": "Oli otya?"},
                      {"role": "assistant", "content": "Gyendi, weebale."}]},
        {"messages": [{"role": "user", "content": "Oli otya?"},
                      {"role": "assistant", "content": "Gyendi, weebale."}]},   # exact duplicate
        {"messages": [{"role": "user", "content": "Weebale nnyo."},
                      {"role": "assistant", "content": "Kale, weebale ggwe."}]},
    ]
    src = tmp_path / "in.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    assert build_dataset.key_of(rows[0]) == build_dataset.key_of(rows[1])
    kept, drops, dups = _dedupe(build_dataset, rows)
    assert len(kept) == 2 and dups == 1


def _dedupe(mod, rows):
    seen, kept, dups = {}, [], 0
    drops = {}
    for rec in rows:
        clean, why = mod.clean_record(rec, min_chars=2, max_turns=20)
        if not clean:
            drops[why] = drops.get(why, 0) + 1
            continue
        k = mod.key_of(clean)
        if k in seen:
            dups += 1
            continue
        seen[k] = clean
        kept.append(clean)
    return kept, drops, dups
