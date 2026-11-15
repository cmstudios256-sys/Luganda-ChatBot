"""Turn books, handbooks, articles and subtitle files into clean text segments.

Formats: .pdf .docx .epub .html/.htm .txt/.md .srt/.vtt .csv/.jsonl

Two rules that matter more than the parsers:

1. **Never silently keep garbage.** A scanned PDF with no text layer yields
   "empty page" markers, not nothing, so you can count what needs OCR.
2. **Keep the locator.** Every segment carries where it came from (page, paragraph,
   subtitle timecode). When a native reviewer says "this sentence is wrong", you need
   to find the page.
"""
from __future__ import annotations

import csv
import json
import re
import zipfile
from dataclasses import dataclass, asdict, field
from pathlib import Path

SUPPORTED = {".pdf", ".docx", ".epub", ".html", ".htm", ".txt", ".md", ".srt", ".vtt", ".csv", ".jsonl"}


@dataclass
class Segment:
    text: str
    source: str            # logical source, e.g. "files:books/oluganda-sseta-1.pdf"
    kind: str              # book | handbook | news | subtitle | transcript | web | unknown
    locator: str = ""      # "p.12" | "para 4" | "00:01:22" | url
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def _clean(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def guess_kind(path: Path) -> str:
    name = path.name.lower()
    if path.suffix.lower() in {".html", ".htm"} and not any(
        w in name for w in ("paper", "news", "bukedde", "newvision", "amawulire")
    ):
        return "web"
    if "srt" in path.suffix or "vtt" in path.suffix:
        return "subtitle"
    if any(w in name for w in ("ebbaluwa", "letter", "emabaluwa")):
        return "letter"
    if any(w in name for w in ("olugero", "story", "ennyimba", "poem", "ekifaananyi")):
        return "literature"
    if any(w in name for w in ("paper", "newspaper", "bukedde", "newvision", "amawulire", "news")):
        return "news"
    if any(w in name for w in ("handbook", "guide", "manual", "kabalaga", "bulagirizi")):
        return "handbook"
    if any(w in name for w in ("book", "textbook", "kitabo", "essomo", "sseta", "novel")):
        return "book"
    return "unknown"


# --------------------------------------------------------------------------- #
# Per-format extractors
# --------------------------------------------------------------------------- #
def from_pdf(path: Path) -> list[Segment]:
    try:
        from pypdf import PdfReader  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("pip install pypdf to read PDFs") from exc

    reader = PdfReader(str(path))
    out: list[Segment] = []
    empty = 0
    for i, page in enumerate(reader.pages, 1):
        try:
            text = _clean(page.extract_text() or "")
        except Exception:                      # noqa: BLE001 - a broken page must not kill the run
            text = ""
        if len(text.split()) < 5:
            empty += 1
            out.append(Segment("", f"files:{path.name}", guess_kind(path), f"p.{i}",
                               {"needs_ocr": True, "chars": len(text)}))
            continue
        out.append(Segment(text, f"files:{path.name}", guess_kind(path), f"p.{i}"))
    if empty and empty == len(reader.pages):
        out.append(Segment("", f"files:{path.name}", guess_kind(path), "document",
                           {"warning": "no text layer at all — this is a scan; run OCR before ingesting"}))
    return out


def from_docx(path: Path) -> list[Segment]:
    try:
        import docx  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("pip install python-docx to read .docx") from exc

    document = docx.Document(str(path))
    paras = [_clean(p.text) for p in document.paragraphs]
    joined = "\n\n".join(p for p in paras if p)
    return [Segment(joined, f"files:{path.name}", guess_kind(path), "document")] if joined else []


def from_epub(path: Path) -> list[Segment]:
    out: list[Segment] = []
    with zipfile.ZipFile(path) as zf:
        for name in sorted(n for n in zf.namelist() if n.lower().endswith((".xhtml", ".html", ".htm"))):
            try:
                raw = zf.read(name).decode("utf-8", "replace")
            except Exception:                  # noqa: BLE001
                continue
            text = _clean(html_to_text(raw))
            if text:
                out.append(Segment(text, f"files:{path.name}", guess_kind(path), name))
    return out


def html_to_text(raw: str) -> str:
    """Article extraction: trafilatura if installed, BeautifulSoup otherwise."""
    try:
        import trafilatura  # noqa: PLC0415

        extracted = trafilatura.extract(raw, include_comments=False, include_tables=True,
                                        favor_recall=True)
        if extracted:
            return extracted
    except ImportError:
        pass
    try:
        from bs4 import BeautifulSoup  # noqa: PLC0415

        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()
        return soup.get_text("\n")
    except ImportError:
        return re.sub(r"<[^>]+>", " ", raw)


def from_html(path: Path) -> list[Segment]:
    text = _clean(html_to_text(path.read_text(encoding="utf-8", errors="replace")))
    return [Segment(text, f"files:{path.name}", guess_kind(path), "document")] if text else []


TIMECODE_RE = re.compile(r"(\d{1,2}:\d{2}:\d{2})[,.]\d{3}\s*-->")


def from_subtitles(path: Path, *, block_seconds: int = 120) -> list[Segment]:
    """Subtitles become transcript segments, grouped into ~2-minute blocks so a
    retrieval hit maps to a watchable moment in the show."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines()
    current_time = ""
    buf: list[str] = []
    out: list[Segment] = []
    for line in lines:
        match = TIMECODE_RE.search(line)
        if match:
            current_time = match.group(1)
            continue
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.isdigit() or stripped.upper().startswith("WEBVTT"):
            continue
        if stripped == "[MUSIC]" or stripped.lower().startswith(("subscribe", "follow us")):
            continue
        buf.append(stripped)
        if len(" ".join(buf).split()) > 180:
            out.append(Segment(_clean(" ".join(buf)), f"files:{path.name}", "subtitle", current_time))
            buf = []
    if buf:
        out.append(Segment(_clean(" ".join(buf)), f"files:{path.name}", "subtitle", current_time))
    return [s for s in out if s.text]


def from_plain(path: Path) -> list[Segment]:
    text = _clean(path.read_text(encoding="utf-8", errors="replace"))
    return [Segment(text, f"files:{path.name}", guess_kind(path), "document")] if text else []


def from_jsonl(path: Path) -> list[Segment]:
    out = []
    for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = _clean(str(rec.get("text") or rec.get("content") or rec.get("transcript") or ""))
        if text:
            out.append(Segment(text, f"files:{path.name}", rec.get("kind", guess_kind(path)),
                               rec.get("locator", f"line {i}"), {"verified": rec.get("verified", False)}))
    return out


def from_csv(path: Path) -> list[Segment]:
    out = []
    with path.open(encoding="utf-8", errors="replace", newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh), 1):
            text = _clean(" ".join(str(v) for v in row.values() if v))
            if text:
                out.append(Segment(text, f"files:{path.name}", guess_kind(path), f"row {i}"))
    return out


EXTRACTORS = {
    ".pdf": from_pdf, ".docx": from_docx, ".epub": from_epub,
    ".html": from_html, ".htm": from_html,
    ".txt": from_plain, ".md": from_plain,
    ".srt": from_subtitles, ".vtt": from_subtitles,
    ".jsonl": from_jsonl, ".csv": from_csv,
}


def extract(path: Path) -> list[Segment]:
    fn = EXTRACTORS.get(path.suffix.lower())
    if not fn:
        raise RuntimeError(f"unsupported format {path.suffix} ({path.name}); supported: {sorted(SUPPORTED)}")
    return fn(path)


def split_big_segment(seg: Segment, max_words: int = 900) -> list[Segment]:
    """Long documents are split on paragraph boundaries so retrieval has granular chunks."""
    if len(seg.text.split()) <= max_words:
        return [seg]
    chunks, buf, words = [], [], 0
    for para in seg.text.split("\n\n"):
        count = len(para.split())
        if words + count > max_words and buf:
            chunks.append(Segment("\n\n".join(buf), seg.source, seg.kind, seg.locator, dict(seg.meta)))
            buf, words = [para], count
        else:
            buf.append(para)
            words += count
    if buf:
        chunks.append(Segment("\n\n".join(buf), seg.source, seg.kind, seg.locator, dict(seg.meta)))
    return chunks
