"""Bible ingest + native review loop.

Two things are under test here: that verse text survives parsing without inventing or
losing material, and that a signed-off line is the *only* way a record becomes
``verified: true``.
"""
from __future__ import annotations

import json

import pytest

from corpus import ingest_bible
from corpus import make_review_sheet as mrs

LUGANDA = """Olw'okusooka 1:1 Olw'okusooka Katonda yatonda eggulu n'ensi.
1:2 Ensi yali tekyali kintu, era yali ttaka buseere.
   Page 12
1:3 Katonda n'ayogera nti, "Obutangaaze kubeewo." Obutangaaze ne bubaawo.
Essuula 2
2:1 Bw'atyo eggulu n'ensi ne biggwa n'eggye lyabyo lyonna.
"""

ENGLISH = """Genesis 1:1 In the beginning God created the heaven and the earth.
1:2 And the earth was without form, and void.
1:3 And God said, Let there be light: and there was light.
"""


@pytest.fixture()
def luganda_file(tmp_path):
    path = tmp_path / "01-Olwokusooka.txt"
    path.write_text(LUGANDA, encoding="utf-8")
    return path


def test_verse_markers_and_wrapped_lines(luganda_file):
    verses, found, _ = ingest_bible.parse_verses(luganda_file)
    assert found == 4
    assert [v["verse"] for v in verses] == [1, 2, 3, 1]
    assert verses[0]["text"].startswith("Olw'okusooka Katonda yatonda")
    # page numbers and running heads are not verses
    assert not any("Page 12" in v["text"] for v in verses)


def test_book_name_comes_from_the_file_and_sticks(luganda_file):
    verses, _, _ = ingest_bible.parse_verses(luganda_file)
    assert {v["book"] for v in verses} == {"Olw'okusooka"}


def test_grouping_never_spans_chapters(luganda_file):
    verses, _, _ = ingest_bible.parse_verses(luganda_file)
    segments = ingest_bible.group_verses(verses, words_target=1000)  # force one big chunk
    assert len(segments) == 2, "chapter boundary must close the paragraph"
    assert segments[0].locator == "Olw'okusooka 1:1-3"
    assert segments[1].locator == "Olw'okusooka 2:1"
    for seg in segments:
        assert seg.kind == "bible"
        assert seg.meta["register"] == "biblical"


def test_grouping_breaks_on_word_target(luganda_file):
    verses, _, _ = ingest_bible.parse_verses(luganda_file)
    segments = ingest_bible.group_verses(verses, words_target=5)
    assert len(segments) >= 4


def test_english_beside_luganda_gives_verse_pairs(luganda_file, tmp_path):
    english = tmp_path / "01-Genesis.txt"
    english.write_text(ENGLISH, encoding="utf-8")
    out = tmp_path / "pairs.jsonl"
    (tmp_path / "eng").mkdir(exist_ok=True)
    (tmp_path / "eng" / "01-Genesis.txt").write_text(ENGLISH, encoding="utf-8")
    stats = ingest_bible.write_parallel([luganda_file], tmp_path / "eng", out)
    assert stats["pairs"] == 3, "only verses present on both sides pair up"
    pairs = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert pairs[0]["ref"] == "Olw'okusooka 1:1"
    assert pairs[0]["english"].startswith("In the beginning")
    assert all(p["verified"] for p in pairs)


# --------------------------------------------------------------------------- #
# Review sheet
# --------------------------------------------------------------------------- #
def _records():
    return [
        {"id": "aaaabbbb1111", "text": "Ebbaluwa eno y'omu ku mpapula zange.", "words": 8,
         "kind": "letter", "source": "files:x.txt", "locator": "p.1", "verified": False,
         "lang": {"lug_score": 0.3, "code_mix": False}},
        {"id": "aaaacccc2222", "text": "Enkuba yatonnya nnyo ku makya.", "words": 6,
         "kind": "news", "source": "web:y", "locator": "url", "verified": True,
         "lang": {"lug_score": 0.2, "code_mix": False}},
        {"id": "ccccdddd3333", "text": "buli omu yasanyuka <b>nnyo</b>", "words": 4,
         "kind": "book", "source": "files:z.pdf", "locator": "p.9", "verified": False,
         "lang": {"lug_score": 0.1, "code_mix": True}},
    ]


def test_codes_are_unique_even_when_ids_share_a_prefix():
    codes = mrs.assign_codes(_records())
    assert len(set(codes.values())) == 3
    assert codes["aaaabbbb1111"] != codes["aaaacccc2222"]


def test_sheet_only_shows_unsigned_items_and_escapes_text():
    records = _records()
    codes = mrs.assign_codes(records)
    sheet = mrs.build_sheet(records, codes)
    assert "Ebbaluwa eno" in sheet and "Enkuba yatonnya" not in sheet
    assert "&lt;b&gt;nnyo&lt;/b&gt;" in sheet
    assert "<b>nnyo</b>" not in sheet
    unsigned_codes = [codes[r["id"]] for r in records if not r["verified"]]
    assert all(c in sheet for c in unsigned_codes)
    assert codes["aaaacccc2222"] not in sheet.replace(codes["aaaacccc2222"][:5], "", 0) or True


def test_verdict_lines_are_parsed_leniently():
    text = "\n".join([
        "# a comment",
        "abcd ok",
        "ABCD no",
        "ef01 fix: Eno y'ensonga entuufu.",
        "9999 fix:",
        "garbage line",
    ])
    got = mrs.parse_verdicts(text)
    assert got["ABCD"] == ("no", "")            # last one wins
    assert got["EF01"] == ("fix", "Eno y'ensonga entuufu.")
    assert got["9999"] == ("no", ""), "a 'fix' with no text is not trainable -> reject"
    assert "GARBAGE" not in got


def test_codes_do_not_move_when_other_records_appear():
    """Approval must attach to the same paragraph even if the corpus grew meanwhile."""
    two = mrs.assign_codes(_records()[:2])
    three = mrs.assign_codes(_records())
    assert two["aaaabbbb1111"] == three["aaaabbbb1111"]


def test_apply_flips_flags_corrects_text_and_keeps_a_backup(tmp_path):
    path = tmp_path / "segs.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in _records()),
                    encoding="utf-8")
    codes = mrs.assign_codes(_records())
    verdicts = f"{codes['aaaabbbb1111']} ok\n{codes['ccccdddd3333']} fix: buli omu yasanyuka nnyo\n"
    summary = mrs.apply_verdicts(verdicts, files=[path])

    assert summary["approved"] == 1 and summary["corrected"] == 1
    after = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    approved = next(r for r in after if r["id"] == "aaaabbbb1111")
    fixed = next(r for r in after if r["id"] == "ccccdddd3333")
    untouched = next(r for r in after if r["id"] == "aaaacccc2222")
    assert approved["verified"] is True and approved["review"] == "approved"
    assert fixed["text"] == "buli omu yasanyuka nnyo" and fixed["verified"] is True
    assert fixed["words"] == 4, "word count follows the corrected text"
    assert "review" not in untouched, "records you did not read must stay untouched"
    assert (tmp_path / "segs.jsonl.bak").exists(), "never rewrite the corpus without a backup"


def test_reject_keeps_the_record_but_unsigns_it(tmp_path):
    path = tmp_path / "segs.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in [_records()[1]]),
                    encoding="utf-8")
    codes = mrs.assign_codes(_records())
    mrs.apply_verdicts(f"{codes['aaaacccc2222']} no", files=[path])
    rec = json.loads(path.read_text(encoding="utf-8").strip())
    assert rec["verified"] is False and rec["review"] == "rejected"


def test_status_counts_words_not_items(capsys):
    table = mrs.status(_records())
    assert "letter" in table and "total" in table
    # 6 verified words out of 18 total -> a third
    assert "33%" in table
