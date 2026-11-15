#!/usr/bin/env python3
"""TV shows and TikTok → transcripts.

Spoken Luganda is where the *real* register lives: how people actually greet, argue,
joke, price things. It is also the noisiest data you can collect, so it is handled
differently from text: every segment is tagged `modality: spoken`, `verified: false`,
and kept out of nothing — but never trusted for spelling.

    # you already have subtitles (best case — no ASR error at all)
    python -m corpus.ingest_media --transcript shows/agataliiko.srt --kind tv --out media

    # a local video/audio file, transcribed with a local Whisper
    python -m corpus.ingest_media --file shows/ep1.mp4 --asr-command \
        "faster-whisper --model large-v3 --language lg --output_format srt {out} {input}"

    # a URL (TikTok/YouTube) — download audio with yt-dlp, then transcribe
    python -m corpus.ingest_media --url https://www.tiktok.com/@x/video/123 --platform tiktok --dry-run

    # Hugging Face Inference API with Sunbird's Luganda Whisper (needs HF_TOKEN)
    python -m corpus.ingest_media --file clip.m4a --asr hf --out media

Rights: a TikTok clip is somebody's work, and often contains music you cannot licence.
Use it to *study* register; publishing a model trained on it, or shipping transcripts,
is a different decision. Prefer: shows you have permission for, your own recordings,
or the openly licensed speech corpora (SALT, WAXAL) listed in CORPUS-PLAN.md.
"""
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import httpx

from .pipeline import OUT, print_stats, process, write_segments
from .text_extract import Segment, extract, from_subtitles

MEDIA_DIR = OUT / "media"
HF_ASR_MODEL = "Sunbird/asr-whisper-large-v3-salt"


def download_audio(url: str, *, platform: str = "") -> Path:
    """yt-dlp pulls the audio track. Never re-upload what comes out of here."""
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    template = str(MEDIA_DIR / "%(extractor)s-%(id)s.%(ext)s")
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--no-playlist", "--extract-audio", "--audio-format", "m4a",
        "--audio-quality", "5", "-o", template, url,
    ]
    print("  $", " ".join(shlex.quote(c) for c in cmd))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # noqa: S603
    except FileNotFoundError as exc:
        raise SystemExit("yt-dlp is not installed: pip install yt-dlp (ffmpeg also recommended)") from exc
    if proc.returncode != 0:
        raise SystemExit(f"download failed:\n{proc.stderr[-1500:]}")
    matches = sorted(MEDIA_DIR.glob("*.m4a")) + sorted(MEDIA_DIR.glob("*.mp3")) + sorted(MEDIA_DIR.glob("*.webm"))
    if not matches:
        raise SystemExit("download reported success but no audio file was found")
    path = max(matches, key=lambda p: p.stat().st_mtime)
    print(f"  → {path}  ({path.stat().st_size/1e6:.1f} MB, platform={platform or 'unknown'})")
    return path


def transcribe_hf(audio: Path, model: str = HF_ASR_MODEL) -> str:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set — needed for the Hugging Face ASR API")
    if audio.stat().st_size > 25 * 1024 * 1024:
        raise SystemExit("over 25 MB: split the audio (ffmpeg -f segment) or run Whisper locally")
    resp = httpx.post(
        f"https://api-inference.huggingface.co/models/{model}",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "audio/m4a"},
        content=audio.read_bytes(),
        timeout=600,
    )
    if resp.status_code >= 400:
        raise SystemExit(f"HF ASR failed: HTTP {resp.status_code} {resp.text[:300]}")
    data = resp.json()
    if isinstance(data, dict):
        return str(data.get("text", ""))
    return " ".join(str(d.get("text", "")) for d in data)


def transcribe_command(audio: Path, template: str) -> str:
    """Run any local ASR CLI. `{input}` and `{out}` are substituted."""
    out_srt = audio.with_suffix(".srt")
    cmd = template.replace("{input}", shlex.quote(str(audio))).replace("{out}", shlex.quote(str(out_srt)))
    print("  $", cmd)
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=7200)  # noqa: S602
    if proc.returncode != 0:
        raise SystemExit(f"ASR command failed:\n{proc.stderr[-1500:]}")
    if out_srt.exists():
        return out_srt.read_text(encoding="utf-8", errors="replace")
    stdout = proc.stdout.strip()
    if not stdout:
        raise SystemExit("ASR produced neither an .srt nor stdout text")
    return stdout


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="TikTok/YouTube/etc URL (downloaded with yt-dlp)")
    src.add_argument("--file", help="local audio or video file")
    src.add_argument("--transcript", help="existing .srt/.vtt/.txt transcript — no ASR at all")
    ap.add_argument("--platform", default="", help="tiktok | youtube | tv | radio (for provenance)")
    ap.add_argument("--kind", default="transcript", help="tv | tiktok | radio | transcript")
    ap.add_argument("--asr", choices=["hf", "none"], default="hf", help="which ASR backend to use")
    ap.add_argument("--asr-command", default="", help="local ASR command template using {input} and {out}")
    ap.add_argument("--model", default=HF_ASR_MODEL)
    ap.add_argument("--out", default="media")
    ap.add_argument("--min-words", type=int, default=40,
                    help="spoken segments need more words to be worth keeping than articles")
    ap.add_argument("--keep-mixed", action="store_true", default=True)
    ap.add_argument("--drop-mixed", dest="keep_mixed", action="store_false")
    ap.add_argument("--dry-run", action="store_true", help="show the plan and the commands, do nothing")
    args = ap.parse_args()

    platform = args.platform or (urlparse(args.url).netloc.replace("www.", "").split(".")[0] if args.url else args.kind)

    # ---- 1. get a transcript (or subtitles) ----
    segments: list[Segment] = []
    if args.transcript:
        path = Path(args.transcript)
        if not path.exists():
            raise SystemExit(f"not found: {path}")
        if path.suffix.lower() in {".srt", ".vtt"}:
            segments = from_subtitles(path)
        else:
            segments = extract(path)
        for seg in segments:
            seg.kind = args.kind
            seg.source = f"media:{platform}:{path.name}"
            seg.meta.update({"modality": "spoken", "platform": platform, "verified": False,
                             "rights": "check before publishing"})
        print(f"{len(segments)} segment(s) from {path.name}")

    else:
        audio = Path(args.file) if args.file else None
        if args.dry_run:
            print("DRY RUN — this is what would happen:")
            if not audio:
                print(f"  1. yt-dlp downloads the audio of {args.url}")
                audio = MEDIA_DIR / f"{platform}-<id>.m4a"
                print(f"     → {audio}")
            else:
                print(f"  1. use local media {audio}")
            if args.asr_command:
                print(f"  2. run: {args.asr_command.replace('{input}', str(audio)).replace('{out}', str(audio.with_suffix('.srt')))}")
            elif args.asr == "hf":
                print(f"  2. POST the audio to Hugging Face Inference API, model {args.model}")
                print("     (requires HF_TOKEN, audio under 25 MB)")
            else:
                print("  2. no ASR backend chosen — nothing to do")
            print(f"  3. filter to Luganda segments (>= {args.min_words} words), write corpus/out/segments/{args.out}.jsonl")
            return 0
        if not audio:
            audio = download_audio(args.url, platform=platform)
        if not audio.exists():
            raise SystemExit(f"not found: {audio}")

        if args.asr_command:
            raw = transcribe_command(audio, args.asr_command)
        elif args.asr == "hf":
            print(f"  transcribing with {args.model} …")
            raw = transcribe_hf(audio, args.model)
        else:
            raise SystemExit("Choose --asr-command (local Whisper) or --asr hf, or pass --transcript")

        tmp = audio.with_suffix(".asr.srt")
        tmp.write_text(raw, encoding="utf-8")
        segments = from_subtitles(tmp) if "-->" in raw else extract(tmp)
        for seg in segments:
            seg.kind = args.kind
            seg.source = f"media:{platform}:{audio.name}"
            seg.meta.update({"modality": "spoken", "platform": platform, "verified": False,
                             "asr_model": args.model if args.asr == "hf" else "local-command",
                             "asr_unverified": True, "rights": "check before publishing"})
        print(f"{len(segments)} transcribed segment(s)")

    kept, stats = process(segments, min_words=args.min_words, keep_mixed=args.keep_mixed)
    stats["sr_platform_" + platform] = len(kept)
    path = write_segments(args.out, kept)
    print_stats("Media transcript result", stats, path)
    if kept:
        print("\nThese transcripts contain ASR errors. Use them for register and vocabulary,")
        print("not as a spelling reference: mark them `verified: false` until a human listens.")
    return 0 if kept else 1


if __name__ == "__main__":
    sys.exit(main())
