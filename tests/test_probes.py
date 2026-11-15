"""The conversation probes: the machine-checkable half of 'is it conversational yet?'"""
from __future__ import annotations

from finetune import probe_check as pc


def test_probe_set_is_well_formed():
    probes = pc.load_probes()
    assert len(probes) >= 25
    ids = [p["id"] for p in probes]
    assert len(ids) == len(set(ids)), "duplicate probe ids"
    for probe in probes:
        assert probe["prompt"].strip()
        assert probe["tests"].strip(), f"{probe['id']}: say what it is testing"
        # honesty: these were drafted by the assistant and corrected by nobody yet
        assert probe["verified"] is False


def test_clock_probes_forbid_am_pm():
    clock = [p for p in pc.load_probes() if "clock" in p["tests"] or "6-to-6" in p["tests"]]
    assert clock
    assert all("AM" in p.get("must_not_include", []) for p in clock)


# --------------------------------------------------------------------------- #
# English leakage
# --------------------------------------------------------------------------- #
def test_luganda_replies_are_not_accused_of_being_english():
    for reply in [
        "Ssaawa emu ey'oku makya, era olunaku lwe lusuze lwali Olw'okutaano.",
        "Ntegedde bulungi, mukwano gwange.",
        "Wadde twatuuka ku ssomero nga tukooye nnyo, luno lwali lukendo lulungi.",
    ]:
        assert pc.english_leak(reply) == [], reply


def test_english_inside_a_luganda_reply_is_caught():
    assert "think" in pc.english_leak("I think kino kirungi")
    assert pc.english_leak("Tewali API key eteekeddwa")


# --------------------------------------------------------------------------- #
# Reply checks
# --------------------------------------------------------------------------- #
def test_a_good_reply_has_no_problems():
    probe = {"id": "c05", "max_words": 30}
    assert pc.check_reply(probe, "Kati ssaawa emu ey'oku makya.") == []


def test_written_clock_is_a_failure_not_a_warning():
    problems = pc.check_reply({"id": "c05"}, "Kati ssaawa 7:00 AM.")
    assert [p["check"] for p in problems] == ["clock-format"]
    assert problems[0]["level"] == "fail"


def test_missing_and_forbidden_content_are_both_enforced():
    probe = {"id": "c15", "must_include": ["Kiyira"], "must_not_include": ["nnyanja"]}
    checks = {p["check"] for p in pc.check_reply(probe, "Yee, ennyanja ya Jinja nene.")}
    assert checks == {"must-include", "must-not-include"}


def test_length_budget_warns_then_fails():
    probe = {"id": "c28", "max_words": 20}
    warn = pc.check_reply(probe, " ".join(["kigambo"] * 25))
    fail = pc.check_reply(probe, " ".join(["kigambo"] * 40))
    assert [p["level"] for p in warn] == ["warn"]
    assert [p["level"] for p in fail] == ["fail"]


def test_empty_reply_fails_loudly():
    assert pc.check_reply({"id": "c01"}, "   ")[0]["check"] == "empty"


def test_false_certainty_about_prices_is_flagged():
    probe = {"id": "c08"}
    problems = pc.check_reply(probe, "Ssente za mmere eno ddala kikumi.")
    assert any(p["check"] == "false-certainty" for p in problems)


def test_hedged_answer_about_prices_is_fine():
    probe = {"id": "c08"}
    problems = pc.check_reply(probe, "Simanyi ssente za mmere eno; kyetaagisa okubuuzza omusuubuzi.")
    assert problems == []


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #
def test_summary_separates_clean_warn_and_fail():
    results = [
        {"id": "a", "problems": []},
        {"id": "b", "problems": [{"level": "warn", "check": "long", "detail": ""}]},
        {"id": "c", "problems": [{"level": "fail", "check": "clock-format", "detail": ""},
                                 {"level": "warn", "check": "long", "detail": ""}]},
    ]
    assert pc.summarize(results) == {"probes": 3, "failed": 1, "warned": 1, "clean": 1}
