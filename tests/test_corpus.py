"""Tests for the corpus pipeline: extraction, language filtering, dedupe, robots.

These run offline. Formats that need optional libraries are skipped if the library
is missing, so the suite stays green on a bare machine.
"""
from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from corpus import build_corpus, langid  # noqa: E402
from corpus.pipeline import process  # noqa: E402
from corpus.text_extract import Segment, from_html, from_subtitles, split_big_segment  # noqa: E402

LUGANDA = (
    "Gavumenti etangazza nti ejja kuzimba oluguudo oluva mu Kampala okutuuka mu Jinja mu myezi "
    "musanvu egijja. Minisita w'ebyentambula yagambye nti ssente zonna ziteekeddwa ku bbali era "
    "nti abakola omulimu bajja kutandika mangu."
)
ENGLISH = (
    "The government has announced that it will rebuild the road from Kampala to Jinja within "
    "seven months. The minister of works said all funds have been allocated and that workers "
    "will begin soon."
)
SWAHILI = (
    "Serikali imetangaza kwamba itajenga upya barabara kutoka Kampala hadi Jinja ndani ya miezi "
    "saba ijayo. Waziri wa ujenzi alisema fedha zote zimetengwa na wafanyakazi wataanza hivi karibuni."
)
CODE_MIXED = (
    "Bino byonna boss, the deadline yaffe is Friday kubanga clients baagala delivery before the "
    "end of this month. Kale tulina okukola extra hours ku Lwokubiri ne ku Lwokusatu."
)


# --------------------------------------------------------------------------- #
# Language filtering
# --------------------------------------------------------------------------- #
def test_langid_separates_luganda_from_english():
    lug = langid.classify(LUGANDA)
    eng = langid.classify(ENGLISH)
    assert lug.decision == "lug" and lug.lug_score > eng.lug_score
    assert eng.decision == "eng"


def test_langid_flags_swahili_as_other_not_luganda():
    assert langid.classify(SWAHILI).decision == "other"


def test_langid_flags_code_mixing():
    result = langid.classify(CODE_MIXED)
    assert result.decision == "mixed"
    assert result.code_mix is True


def test_langid_needs_enough_text():
    assert langid.classify("Oli otya?").decision == "unknown"


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def test_html_extraction_drops_navigation():
    html = ("<html><body><nav>Ewaka Amawulire</nav><article><p>" + LUGANDA + "</p></article>"
            "<script>var x = 1;</script><footer>© Bukedde</footer></body></html>")
    segments = from_html_from_string(html)
    assert LUGANDA.split()[0] in segments
    assert "var x" not in segments
    assert "Ewaka Amawulire" not in segments


def from_html_from_string(html: str) -> str:
    from corpus.text_extract import html_to_text

    return html_to_text(html)


def test_subtitles_become_timecoded_segments(tmp_path):
    srt = tmp_path / "show.srt"
    srt.write_text(
        "1\n00:00:04,000 --> 00:00:09,000\nOsiibye otya nno, mukwano?\n\n"
        "2\n00:00:09,500 --> 00:00:15,000\nNsibye bulungi. Nzize jjo ku ggulo.\n",
        encoding="utf-8",
    )
    segments = from_subtitles(srt)
    assert segments and segments[0].locator.count(":") == 2
    assert "Osiibye" in segments[0].text


def test_pdf_text_and_scan_detection(tmp_path):
    fpdf = pytest.importorskip("fpdf", reason="pip install fpdf2 to test PDF extraction")
    from corpus.text_extract import from_pdf

    good = tmp_path / "book.pdf"
    pdf = fpdf.FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 6, LUGANDA)
    pdf.output(str(good))
    segments = from_pdf(good)
    assert any(LUGANDA.split()[0] in s.text for s in segments)

    blank = tmp_path / "scan.pdf"
    empty = fpdf.FPDF()
    empty.add_page()
    empty.output(str(blank))
    scanned = from_pdf(blank)
    assert any(s.meta.get("needs_ocr") or "no text layer" in str(s.meta) for s in scanned)


def test_docx_extraction(tmp_path):
    docx = pytest.importorskip("docx", reason="pip install python-docx")
    from corpus.text_extract import from_docx

    path = tmp_path / "handbook.docx"
    document = docx.Document()
    document.add_paragraph(LUGANDA)
    document.add_paragraph(LUGANDA.replace("Gavumenti", "Abalimi"))
    document.save(str(path))
    segments = from_docx(path)
    assert segments and "Gavumenti" in segments[0].text


def test_long_segments_are_split():
    seg = Segment("\n\n".join([LUGANDA] * 30), "files:x", "book", "p.1")
    chunks = split_big_segment(seg, max_words=200)
    assert len(chunks) > 1
    assert all(len(c.text.split()) <= 400 for c in chunks)


# --------------------------------------------------------------------------- #
# Filtering + shaping
# --------------------------------------------------------------------------- #
def test_process_keeps_luganda_and_drops_english():
    segments = [
        Segment(LUGANDA, "files:a", "news", "p.1"),
        Segment(ENGLISH, "files:b", "news", "p.1"),
        Segment(SWAHILI, "files:c", "news", "p.1"),
    ]
    kept, stats = process(segments, min_words=10)
    assert len(kept) == 1
    assert kept[0]["source"] == "files:a"
    assert kept[0]["verified"] is False
    assert stats["rejected_english"] == 1
    assert stats["rejected_not_luganda"] == 1


def test_process_keeps_code_mixed_with_a_flag():
    kept, stats = process([Segment(CODE_MIXED * 2, "files:m", "transcript", "00:01")], min_words=10)
    assert kept and kept[0]["lang"]["code_mix"] is True
    assert stats["kept_code_mixed"] == 1


def test_boilerplate_is_removed_but_content_survives():
    repeated_header = "Bukedde Online — Amawulire"
    # Each page has the same running head but different body copy, like a real newspaper.
    pages = [
        "Gavumenti etangazza nti ejja kuzimba oluguudo oluva mu Kampala okutuuka mu Jinja.",
        "Essomero ly'e Masaka likyalira abayizi bangi nnyo buli mwaka era abazadde basanyuse.",
        "Abalimi bafuna obuyambi okuva mu gavumenti okusobola okulima emmwanyi bulungi.",
        "Eddwaliro ly'e Mulago likyatira abalwadde bangi buli lunaku mu kiseera kino.",
    ]
    records = [
        {"id": f"r{i}", "text": "\n".join([repeated_header, page]), "words": 100,
         "kind": "news", "source": f"files:p{i}", "modality": "text", "lang": {"code_mix": False}}
        for i, page in enumerate(pages)
    ]
    stripped, stats = build_corpus.strip_repeated_paragraphs(records)
    assert stats["boilerplate_units_removed"] > 0
    assert stripped, "content must survive"
    joined = " ".join(r["text"] for r in stripped)
    assert "Gavumenti" in joined, "body copy must survive"
    assert joined.count(repeated_header) <= 1, "the running head should not survive 4 times"


def test_exact_duplicates_are_collapsed():
    records = [
        {"id": "a", "text": LUGANDA, "words": 40, "kind": "news", "source": "s1", "modality": "text"},
        {"id": "a", "text": LUGANDA, "words": 40, "kind": "news", "source": "s1", "modality": "text"},
        {"id": "b", "text": ENGLISH, "words": 30, "kind": "news", "source": "s2", "modality": "text"},
    ]
    kept, stats = build_corpus.dedupe(records)
    assert len(kept) == 2
    assert stats["dropped_exact_duplicate"] == 1


def test_retrieval_docs_group_by_source_and_locator():
    records = [
        {"id": "1", "text": LUGANDA, "words": 40, "kind": "news", "source": "files:a.pdf",
         "locator": "p.1", "modality": "text", "verified": False},
        {"id": "2", "text": LUGANDA[:80], "words": 12, "kind": "news", "source": "files:a.pdf",
         "locator": "p.1", "modality": "text", "verified": False},
        {"id": "3", "text": ENGLISH, "words": 30, "kind": "web", "source": "web:x", "locator": "u",
         "modality": "text", "verified": False},
    ]
    docs = build_corpus.retrieval_docs(records)
    assert len(docs) == 2
    assert all(set(d) >= {"title", "text", "source"} for d in docs)


def test_cpt_skips_unverified_spoken_transcripts():
    records = [
        {"id": "1", "text": LUGANDA * 6, "words": 300, "kind": "book", "source": "files:b",
         "modality": "text", "verified": False},
        {"id": "2", "text": LUGANDA * 6, "words": 300, "kind": "tv", "source": "media:tv",
         "modality": "spoken", "verified": False},
    ]
    chunks = build_corpus.cpt_chunks(records, words_per_chunk=200)
    assert chunks, "text content should become CPT chunks"
    assert all("Osiibye" not in c for c in chunks), "ASR transcripts must not enter CPT text"


# --------------------------------------------------------------------------- #
# robots.txt compliance
# --------------------------------------------------------------------------- #
class _RobotsHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence
        pass

    def do_GET(self):
        if self.path == "/robots.txt":
            body = "User-agent: *\nDisallow: /news/\nCrawl-delay: 1\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(body.encode())
        else:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"<html><body><p>hi</p></body></html>")

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()


def test_robots_txt_is_obeyed():
    from corpus.ingest_web import robots_for

    server = HTTPServer(("127.0.0.1", 0), _RobotsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        import httpx

        with httpx.Client() as client:
            cache: dict = {}
            parser = robots_for(client, f"{base}/news/story.html", cache)
            assert parser.can_fetch("YogeraCorpusBot/1.0", f"{base}/news/story.html") is False
            assert parser.can_fetch("YogeraCorpusBot/1.0", f"{base}/page/ok.html") is True
            # cache must not refetch per URL
            assert robots_for(client, f"{base}/page/other.html", cache) is parser
    finally:
        server.shutdown()


# --------------------------------------------------------------------------- #
# Bulk loading into the app
# --------------------------------------------------------------------------- #
def test_bulk_knowledge_endpoint(client):
    docs = [{"title": f"Corpus doc {i}",
             "text": f"{LUGANDA} Ekitundu kya {i} kiraga nti abantu baagala amakuru.", "source": "corpus:test"}
            for i in range(3)]
    r = client.post("/api/knowledge/bulk", json={"documents": docs})
    assert r.status_code == 200
    body = r.json()
    assert body["added"] == 3
    assert body["chunks_added"] >= 3
    hit = client.get("/api/knowledge/search", params={"q": "amakuru ku luguudo"}).json()
    assert hit["documents"], "bulk-loaded documents must be retrievable"


# --------------------------------------------------------------------------- #
# Register caps: the Bible must not become the only voice the model has
# --------------------------------------------------------------------------- #
def _reg_records(biblical=20, other=5, words=100):
    recs = [{"id": f"b{i:02d}", "words": words, "register": "biblical",
             "source": "bible:Olw'okusooka", "locator": str(i), "text": "x"} for i in range(biblical)]
    recs += [{"id": f"o{i:02d}", "words": words, "source": "files:ebbaluwa", "locator": str(i),
              "text": "x"} for i in range(other)]
    return recs


def test_biblical_share_is_capped():
    from corpus.build_corpus import cap_registers

    kept, report = cap_registers(_reg_records(), {"biblical": 0.25})
    bible_words = sum(r["words"] for r in kept if r.get("register") == "biblical")
    total = sum(r["words"] for r in kept)
    assert bible_words < 2000, "no capping happened"
    assert bible_words / total <= 0.30, f"still {bible_words / total:.0%} biblical"
    assert report["biblical"]["dropped_records"] > 0


def test_capping_is_deterministic():
    from corpus.build_corpus import cap_registers

    a, _ = cap_registers(_reg_records(), {"biblical": 0.25})
    b, _ = cap_registers(_reg_records(), {"biblical": 0.25})
    assert [r["id"] for r in a] == [r["id"] for r in b], "same corpus must give the same mix"


def test_small_register_is_left_alone():
    from corpus.build_corpus import cap_registers

    kept, report = cap_registers(_reg_records(biblical=1, other=20), {"biblical": 0.25})
    assert len(kept) == 21
    assert report["biblical"]["dropped_records"] == 0


def test_cap_is_skipped_with_a_warning_when_it_is_the_whole_corpus():
    from corpus.build_corpus import cap_registers

    only_bible = [r for r in _reg_records() if r.get("register")]
    kept, report = cap_registers(only_bible, {"biblical": 0.25})
    assert len(kept) == len(only_bible), "deleting the entire corpus is not a fix"
    assert "warning" in report["biblical"]


def test_no_caps_means_no_changes():
    from corpus.build_corpus import cap_registers

    recs = _reg_records()
    kept, report = cap_registers(recs, {})
    assert kept == recs and report == {}
