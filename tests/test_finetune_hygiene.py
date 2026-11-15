"""Dataset hygiene: cleaning characters, keeping schemas apart, and capping the Bible's share.

Each of these tests exists because the thing it checks was actually wrong in this repo:
`ŋ` in the alpaca import, DPO files counted as broken SFT, and no cap on how much of the
training text one register could take.
"""
from __future__ import annotations

import json

from finetune import clean_dataset as cd
from finetune import ingest_bible_pairs as ibp
from finetune import validate_dataset as vd


# --------------------------------------------------------------------------- #
# Character cleaning
# --------------------------------------------------------------------------- #
def test_eng_and_ny_letters_are_transliterated():
    assert cd.clean_text("o\u014bkola") == "ongkola"      # ŋ -> ng
    assert cd.clean_text("e\u0272a") == "enya"            # ɲ -> ny


def test_curly_quotes_and_zero_width_junk_are_normalised():
    assert cd.clean_text("\u201cOli otya?\u201d") == '"Oli otya?"'
    assert cd.clean_text("Oli\u200b otya") == "Oli otya"
    assert cd.clean_text("  emmere   eno  ") == "emmere eno"


def test_seed_style_answer_is_untouched():
    """No silent 'improvements' to Luganda — only mechanical character fixes."""
    original = "Wadde twatuuka ku ssomero nga tukooye nnyo, luno lwali lukendo lulungi."
    assert cd.clean_text(original) == original


def test_records_without_messages_pass_through_untouched():
    """Eval sets and DPO pairs are not chat turns; a cleaner that eats them is a bug."""
    for rec in ({"prompt": "Ssaawa mmeka?", "reference": "Ssaawa emu."},
                {"prompt": "p", "chosen": "c", "rejected": "r"}):
        fixed, changed = cd.clean_record(rec)
        assert fixed == rec and changed is False


def test_empty_turn_makes_a_record_unusable():
    rec = {"messages": [{"role": "user", "content": "\u200b"},
                        {"role": "assistant", "content": "Yee."}]}
    fixed, _ = cd.clean_record(rec)
    assert fixed is None


def test_record_ending_on_a_user_turn_is_dropped():
    rec = {"messages": [{"role": "user", "content": "Oli otya?"}]}
    fixed, _ = cd.clean_record(rec)
    assert fixed is None


def test_cleaning_writes_a_backup(tmp_path):
    path = tmp_path / "x.jsonl"
    rec = {"messages": [{"role": "user", "content": "o\u014bkola ki?"},
                        {"role": "assistant", "content": "Nkola bulungi."}]}
    path.write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
    stats = cd.process_file(path, write=True)
    assert stats["changed"] == 1
    assert (tmp_path / "x.jsonl.bak").exists()
    cleaned = json.loads(path.read_text(encoding="utf-8").strip())
    assert cleaned["messages"][0]["content"] == "ongkola ki?"


# --------------------------------------------------------------------------- #
# Validation schemas
# --------------------------------------------------------------------------- #
def test_dpo_records_are_validated_on_their_own_terms():
    errors, _ = vd.check_record({"prompt": "Oli otya?", "chosen": "Bulungi.", "rejected": "Hi."})
    assert errors == []


def test_dpo_with_identical_choices_is_an_error():
    errors, _ = vd.check_record({"prompt": "p", "chosen": "same", "rejected": "same"})
    assert any("identical" in e for e in errors)


# --------------------------------------------------------------------------- #
# Bible pairs -> SFT, with a share cap
# --------------------------------------------------------------------------- #
PAIRS = [
    {"luganda": "Katonda yatonda eggulu n'ensi.", "english": "God created the heaven and the earth.",
     "ref": "Olw'okusooka 1:1"},
    {"luganda": "Yee.", "english": "Yes.", "ref": "Olw'okusooka 2:2"},          # too short
    {"luganda": "Ensi yali ttaka buseere era nga tekyali kintu.", "english": "The earth was void.",
     "ref": "Olw'okusooka 1:2"},
]


def test_template_choice_is_deterministic():
    a = ibp.pick_template("Olw'okusooka 1:1", ibp.TO_LUGANDA)
    b = ibp.pick_template("Olw'okusooka 1:1", ibp.TO_LUGANDA)
    assert a == b
    assert "{source}" in a


def test_short_verses_are_skipped():
    records = ibp.build_records(PAIRS, limit=10, min_words=4, both_directions=False)
    assert len(records) == 2
    assert all(r["messages"][1]["content"] != "Yee." for r in records)


def test_pairs_produce_valid_sft_records():
    records = ibp.build_records(PAIRS, limit=10, min_words=4, both_directions=False)
    for rec in records:
        errors, _ = vd.check_record(rec)
        assert errors == [], errors
        assert rec["verified"] is True
        assert rec["register"] == "biblical"


def test_both_directions_double_the_records_but_do_not_teach_english_chat():
    records = ibp.build_records(PAIRS, limit=10, min_words=4, both_directions=True)
    assert len(records) == 4
    assert {r["topic"] for r in records} == {"bible-translation", "bible-translation-en"}


def test_share_cap_keeps_translation_a_minority():
    limit, reason = ibp.allowed_count(existing=100, share=0.20, hard_limit=4000)
    assert limit == 25, "20% share of 100 existing -> 25 new records"
    assert "20%" in reason


def test_zero_existing_records_reports_the_real_problem():
    limit, reason = ibp.allowed_count(existing=0, share=0.20, hard_limit=4000)
    assert limit == 4000
    assert "100% translation" in reason


def test_limit_is_respected():
    many = [{"luganda": f"Olugero luno luli lulungi nnyo era lwa bukuumi {i}.",
             "english": f"This story is good and is example {i}.", "ref": f"X 1:{i}"}
            for i in range(50)]
    records = ibp.build_records(many, limit=7, min_words=4, both_directions=False)
    assert len(records) == 7
